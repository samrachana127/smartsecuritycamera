import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import os
import time
from datetime import datetime
from motion_detector import MotionDetector
from clip_saver import ClipSaver
from video_loader import VideoLoader

# ── Colours ───────────────────────────────────────────────────────
BG_MAIN   = "#faf8f5"
BG_PANEL  = "#f0ece6"
BG_DARK   = "#ddd8d0"
TEXT_DARK = "#2a2018"
TEXT_MID  = "#7a6a50"
TEXT_LITE = "#8a7a60"
G         = "#3a6a4a"
R         = "#9a3a3a"
A         = "#8a6a20"
DEF       = "#e8e4de"
PU        = "#5a3a7a"
BORDER    = "#ddd8d0"

PW = 400   # panel width  (fixed — no flicker)
PH = 210   # panel height (fixed — no flicker)

CLR = {
    "HUMAN":   (58,  106,  74),
    "VEHICLE": (32,  106, 154),
    "MOTION":  (154, 106,  32),
}


# ── Tracker ───────────────────────────────────────────────────────
class Tracker:
    def __init__(self):
        self.tracks  = {}
        self.next_id = 1
        self.MAX_TRAIL = 20
        self.MAX_LOST  = 8

    def _iou(self, a, b):
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        if inter == 0: return 0.0
        union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/max(union,1)

    def update(self, detections):
        matched = set()
        for x1,y1,x2,y2,label in detections:
            cx,cy = (x1+x2)//2,(y1+y2)//2
            best_id,best_sc = None,0.1
            for tid,t in self.tracks.items():
                sc = self._iou((x1,y1,x2,y2),t["box"])
                if sc > best_sc: best_sc,best_id = sc,tid
            if best_id is not None:
                t = self.tracks[best_id]
                t["box"]=(x1,y1,x2,y2); t["label"]=label; t["lost"]=0
                t["trail"].append((cx,cy))
                if len(t["trail"]) > self.MAX_TRAIL: t["trail"].pop(0)
                matched.add(best_id)
            else:
                self.tracks[self.next_id]={"box":(x1,y1,x2,y2),"label":label,
                                            "lost":0,"trail":[(cx,cy)]}
                matched.add(self.next_id); self.next_id+=1
        for tid in list(self.tracks):
            if tid not in matched: self.tracks[tid]["lost"]+=1
        self.tracks={k:v for k,v in self.tracks.items() if v["lost"]<=self.MAX_LOST}
        return self.tracks


# ── Helpers ───────────────────────────────────────────────────────
def classify(x1,y1,x2,y2,webcam,outdoor):
    w,h  = x2-x1,y2-y1
    area = w*h
    ratio= h/max(w,1)
    if webcam:
        if h>80 and ratio>0.5 and 5000<area<200000: return "HUMAN"
        if area>200000 and ratio<0.7:                return "VEHICLE"
    elif outdoor:
        if h>35 and ratio>0.9 and 300<area<100000:  return "HUMAN"
        if area>100000 and ratio<0.7:                return "VEHICLE"
    else:
        if h>40 and ratio>0.8 and 800<area<80000 and y2>240: return "HUMAN"
        if area>80000 and ratio<0.6:                 return "VEHICLE"
    return "MOTION"

def undistort(frame,strength=0.25):
    h,w=frame.shape[:2]
    K=np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],dtype=np.float32)
    d=np.array([-strength,strength*0.3,0,0],dtype=np.float32)
    nK,_=cv2.getOptimalNewCameraMatrix(K,d,(w,h),1)
    return cv2.undistort(frame,K,d,None,nK)

def compress(frame,quality):
    _,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,quality])
    return cv2.imdecode(buf,cv2.IMREAD_COLOR)

def to_imgtk(bgr,w=PW,h=PH):
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    img=Image.fromarray(rgb).resize((w,h),Image.LANCZOS)
    return ImageTk.PhotoImage(image=img)

def mask_imgtk(mask,w=PW,h=PH):
    rgb=cv2.cvtColor(cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2RGB)
    img=Image.fromarray(rgb).resize((w,h),Image.LANCZOS)
    return ImageTk.PhotoImage(image=img)

def merge_boxes(boxes,margin=8):
    if not boxes: return []
    exp=[(x1-margin,y1-margin,x2+margin,y2+margin) for x1,y1,x2,y2 in boxes]
    changed=True
    while changed:
        changed,out,used=False,[],[False]*len(exp)
        for i in range(len(exp)):
            if used[i]: continue
            ax1,ay1,ax2,ay2=exp[i]
            for j in range(i+1,len(exp)):
                if used[j]: continue
                bx1,by1,bx2,by2=exp[j]
                if ax1<bx2 and ax2>bx1 and ay1<by2 and ay2>by1:
                    ax1,ay1=min(ax1,bx1),min(ay1,by1)
                    ax2,ay2=max(ax2,bx2),max(ay2,by2)
                    used[j]=True; changed=True
            out.append((ax1,ay1,ax2,ay2)); used[i]=True
        exp=out
    return [(x1+margin,y1+margin,x2-margin,y2-margin) for x1,y1,x2,y2 in exp]


# ════════════════════════════════════════════════════════════════
#  DASHBOARD
# ════════════════════════════════════════════════════════════════
class Dashboard:
    def __init__(self,root):
        self.root=root
        self.root.title("SafeScope Warehouse Security System ")
        self.root.configure(bg=BG_MAIN)
        self.root.geometry("1350x820")

        self.loader   = None
        self.detector = MotionDetector()
        self.saver    = ClipSaver(output_folder="saved_clips")
        self.tracker  = Tracker()

        # ── Cam2: has its OWN thread + lock so it never touches cam1 ──
        self.loader2      = None
        self.detect2      = MotionDetector()
        self._cam2_lock   = threading.Lock()
        self._cam2_frame  = None   # latest annotated frame from cam2
        self._cam2_live   = None   # latest raw frame from cam2 (for stitch)
        self._cam2_running= False

        # ── Cam1 latest raw frame (for stitch) ────────────────────────
        self._cam1_live   = None

        self.running=False; self.paused=False
        self.source="warehouse.mp4"
        self.webcam=False; self.outdoor=False

        self.n_human=0; self.n_vehicle=0; self.n_motion=0
        self.last_log=0
        self.prev_human=False; self.prev_vehicle=False; self.prev_motion=False

        self.img=[None,None,None,None]   # strong refs to prevent GC flicker

        self.do_track    =tk.BooleanVar(value=True)
        self.do_trails   =tk.BooleanVar(value=True)
        self.do_undistort=tk.BooleanVar(value=False)
        self.do_compress =tk.BooleanVar(value=False)
        self.quality     =tk.IntVar(value=60)

        self._build()

    # ──────────────────────────────────────────────────────────────
    #  BUILD UI
    # ──────────────────────────────────────────────────────────────
    def _build(self):
        tb=tk.Frame(self.root,bg=BG_PANEL)
        tb.pack(fill="x")
        tk.Label(tb,text="SafeScope | Northampton Warehouse",
                 bg=BG_PANEL,fg=TEXT_DARK,
                 font=("Arial",12,"bold")).pack(side="left",padx=12,pady=6)
        self.lbl_status=tk.Label(tb,text="OFFLINE",bg=BG_PANEL,fg=R,
                 font=("Arial",9,"bold"))
        self.lbl_status.pack(side="right",padx=12)
        self.lbl_src=tk.Label(tb,text="Source: warehouse.mp4",
                 bg=BG_PANEL,fg=TEXT_MID,font=("Arial",8))
        self.lbl_src.pack(side="right",padx=6)
        tk.Frame(self.root,bg=BORDER,height=1).pack(fill="x")

        sr=tk.Frame(self.root,bg=BG_MAIN)
        sr.pack(fill="x",padx=8,pady=4)
        self.s_motion =self._stat(sr,"Motion","0")
        self.s_human  =self._stat(sr,"Humans","0")
        self.s_vehicle=self._stat(sr,"Vehicles","0")
        self.s_clips  =self._stat(sr,"Clips","0")
        self.s_status =self._stat(sr,"Status","Idle")
        self.s_cam2   =self._stat(sr,"Cam 2","Off")

        r1=tk.Frame(self.root,bg=BG_MAIN)
        r1.pack(fill="x",padx=8,pady=(2,0))
        for txt,cmd in [("Warehouse",self.load_warehouse),("Outdoor",self.load_outdoor),
                        ("Open Video",self.open_video),("Use Camera",self.use_camera),
                        ("+ Cam 2",self.add_cam2),("Stitch View",self.stitch_view)]:
            self._btn(r1,txt,DEF,TEXT_MID,cmd)

        r2=tk.Frame(self.root,bg=BG_MAIN)
        r2.pack(fill="x",padx=8,pady=(2,0))
        self._btn(r2,"Start",G,"#fff",self.start)
        self._btn(r2,"Pause",DEF,TEXT_MID,self.pause)
        self._btn(r2,"Stop",R,"#fff",self.stop)
        self._btn(r2,"Export Clip",DEF,TEXT_MID,self.export_clip)
        tk.Label(r2,text="  |  Features:",bg=BG_MAIN,fg=TEXT_LITE,
                 font=("Arial",8)).pack(side="left")
        for txt,var in [("Track",self.do_track),("Trails",self.do_trails),
                        ("Undistort",self.do_undistort),("Compress",self.do_compress)]:
            tk.Checkbutton(r2,text=txt,variable=var,bg=BG_MAIN,fg=TEXT_MID,
                           font=("Arial",8),selectcolor=BG_DARK,
                           activebackground=BG_MAIN).pack(side="left",padx=4)
        tk.Label(r2,text="Quality:",bg=BG_MAIN,fg=TEXT_LITE,
                 font=("Arial",8)).pack(side="left",padx=(8,0))
        self.lbl_q=tk.Label(r2,text="60",bg=BG_MAIN,fg=TEXT_DARK,
                 font=("Arial",8,"bold"),width=3)
        self.lbl_q.pack(side="left")
        tk.Scale(r2,from_=10,to=95,orient="horizontal",variable=self.quality,
                 bg=BG_MAIN,fg=TEXT_MID,troughcolor=BG_DARK,
                 highlightthickness=0,showvalue=False,length=90,
                 command=lambda v:self.lbl_q.config(text=str(int(float(v))))).pack(side="left")

        tk.Frame(self.root,bg=BORDER,height=1).pack(fill="x",pady=3)

        self.alert=tk.Label(self.root,text="  Monitoring...",
                 bg=G,fg="#fff",font=("Arial",9,"bold"),anchor="w")
        self.alert.pack(fill="x",padx=8,pady=(0,3))

        main=tk.Frame(self.root,bg=BG_MAIN)
        main.pack(fill="both",expand=True,padx=8,pady=2)

        left=tk.Frame(main,bg=BG_MAIN)
        left.pack(side="left")

        def panel(parent,title,attr):
            f=tk.Frame(parent,bg=BG_MAIN)
            f.pack(side="left",padx=2)
            tk.Label(f,text=title,bg=BG_DARK,fg=TEXT_LITE,
                     font=("Arial",7,"bold"),anchor="w",
                     padx=4,width=PW//7).pack(fill="x")
            lbl=tk.Label(f,bg="#111",width=PW,height=PH)
            lbl.pack()
            setattr(self,attr,lbl)

        top_row=tk.Frame(left,bg=BG_MAIN); top_row.pack(pady=(0,3))
        panel(top_row,"HEATMAP",       "p_heat")
        panel(top_row,"LIVE FEED (CAM-01)","p_live")
        bot_row=tk.Frame(left,bg=BG_MAIN); bot_row.pack()
        panel(bot_row,"DETECTION + TRACKING","p_det")
        panel(bot_row,"MOTION MASK",         "p_mask")

        sb=tk.Frame(main,bg=BG_MAIN,width=220)
        sb.pack(side="left",fill="y",padx=(10,0))
        sb.pack_propagate(False)

        tk.Label(sb,text="INCIDENT LOG",bg=BG_MAIN,fg=TEXT_LITE,
                 font=("Arial",8,"bold")).pack(anchor="w",pady=(0,2))
        self.log=tk.Listbox(sb,bg=BG_PANEL,fg=TEXT_MID,font=("Courier",8),
                 selectbackground=BG_DARK,relief="flat",bd=1,
                 highlightthickness=0,height=9)
        self.log.pack(fill="x")

        tk.Label(sb,text="LIVE COUNT",bg=BG_MAIN,fg=TEXT_LITE,
                 font=("Arial",8,"bold")).pack(anchor="w",pady=(8,2))
        cr=tk.Frame(sb,bg=BG_PANEL); cr.pack(fill="x")
        self.c_human  =self._count(cr,"Humans",  "0",R)
        self.c_vehicle=self._count(cr,"Vehicles","0","#3a5a9a")
        self.c_motion =self._count(cr,"Motion",  "0",A)

        tk.Label(sb,text="ZONE MAP",bg=BG_MAIN,fg=TEXT_LITE,
                 font=("Arial",8,"bold")).pack(anchor="w",pady=(8,2))
        for name,bg,lbl in [("Gate A",R,"ALERT"),("Bay N",G,"CLEAR"),
                             ("Loading",A,"MOTION"),("Perimeter",G,"CLEAR")]:
            row=tk.Frame(sb,bg=BG_PANEL); row.pack(fill="x",pady=1,ipady=2)
            tk.Label(row,text=name,bg=BG_PANEL,fg=TEXT_DARK,
                     font=("Arial",9)).pack(side="left",padx=6)
            tk.Label(row,text=lbl,bg=bg,fg="#fff",
                     font=("Arial",8,"bold"),padx=4).pack(side="right",padx=6)

        tk.Label(sb,text="SAVED CLIPS",bg=BG_MAIN,fg=TEXT_LITE,
                 font=("Arial",8,"bold")).pack(anchor="w",pady=(8,2))
        self.clips=tk.Listbox(sb,bg=BG_PANEL,fg=TEXT_MID,font=("Courier",8),
                 selectbackground=BG_DARK,relief="flat",bd=1,
                 highlightthickness=0,height=7)
        self.clips.pack(fill="x")
        self._refresh_clips()

    def _btn(self,parent,text,bg,fg,cmd):
        tk.Button(parent,text=text,bg=bg,fg=fg,
                  font=("Arial",8,"bold"),padx=8,pady=4,
                  bd=0,relief="flat",cursor="hand2",
                  activebackground=BG_DARK,
                  command=cmd).pack(side="left",padx=2)

    def _stat(self,parent,label,value):
        f=tk.Frame(parent,bg=BG_PANEL,padx=10,pady=4,
                   highlightbackground=BORDER,highlightthickness=1)
        f.pack(side="left",padx=3)
        tk.Label(f,text=label,bg=BG_PANEL,fg=TEXT_LITE,font=("Arial",7)).pack()
        v=tk.Label(f,text=value,bg=BG_PANEL,fg=TEXT_DARK,font=("Arial",14,"bold"))
        v.pack(); return v

    def _count(self,parent,label,value,colour):
        f=tk.Frame(parent,bg=BG_PANEL,padx=4,pady=3)
        f.pack(side="left",expand=True)
        tk.Label(f,text=label,bg=BG_PANEL,fg=TEXT_LITE,font=("Arial",7)).pack()
        v=tk.Label(f,text=value,bg=BG_PANEL,fg=colour,font=("Arial",12,"bold"))
        v.pack(); return v

    def _log(self,msg):
        ts=datetime.now().strftime("%H:%M:%S")
        entry=f"{ts} - {msg}"
        self.root.after(0,lambda e=entry: self.log.insert(0,e))

    def _refresh_clips(self):
        if not os.path.exists("saved_clips"): return
        clips=sorted(os.listdir("saved_clips"),reverse=True)
        self.clips.delete(0,tk.END)
        for c in clips: self.clips.insert(tk.END,c)
        self.s_clips.config(text=str(len(clips)))

    # ──────────────────────────────────────────────────────────────
    #  SOURCE CONTROLS
    # ──────────────────────────────────────────────────────────────
    def _switch(self,source):
        was=self.running; self.running=False; time.sleep(0.15)
        if self.loader: self.loader.release(); self.loader=None
        self.detector=MotionDetector(); self.tracker=Tracker()
        self.source=source
        self.loader=VideoLoader(source=source)
        self.prev_human=self.prev_vehicle=self.prev_motion=False
        if was:
            self.running=True
            threading.Thread(target=self._loop,daemon=True).start()

    def load_warehouse(self):
        self.webcam=self.outdoor=False
        self.lbl_src.config(text="Source: warehouse.mp4")
        self.s_status.config(text="Warehouse")
        self._log("Switched to warehouse")
        self._switch("warehouse.mp4")

    def load_outdoor(self):
        self.webcam=False; self.outdoor=True
        self.lbl_src.config(text="Source: outdoor.mp4")
        self.s_status.config(text="Outdoor")
        self._log("Switched to outdoor")
        self._switch("outdoor.mp4")

    def open_video(self):
        path=filedialog.askopenfilename(
            filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.webcam=self.outdoor=False
            name=os.path.basename(path)
            self.lbl_src.config(text=f"Source: {name}")
            self.s_status.config(text="Video")
            self._log(f"Opened: {name}")
            self._switch(path)

    def use_camera(self):
        self.webcam=True; self.outdoor=False
        self.lbl_src.config(text="Source: webcam")
        self.s_status.config(text="Camera")
        self._log("Switched to webcam")
        self._switch(0)

    # ──────────────────────────────────────────────────────────────
    #  CAM 2  — completely independent thread, never touches cam1
    # ──────────────────────────────────────────────────────────────
    def add_cam2(self):
        win=tk.Toplevel(self.root)
        win.title("Add Second Camera")
        win.configure(bg=BG_MAIN)
        win.geometry("300x120")
        win.resizable(False,False)
        tk.Label(win,text="Choose second source:",
                 bg=BG_MAIN,fg=TEXT_DARK,
                 font=("Arial",10,"bold")).pack(pady=12)
        row=tk.Frame(win,bg=BG_MAIN); row.pack()
        def pick_file():
            p=filedialog.askopenfilename(
                filetypes=[("Video","*.mp4 *.avi *.mov")])
            if p: self._set_cam2(p,os.path.basename(p)); win.destroy()
        def pick_webcam():
            self._set_cam2(1,"webcam-1"); win.destroy()
        self._btn(row,"Video File",DEF,TEXT_MID,pick_file)
        self._btn(row,"Webcam 1",DEF,PU,pick_webcam)

    def _set_cam2(self,source,name):
        # Stop existing cam2 thread first
        self._cam2_running=False
        time.sleep(0.2)
        if self.loader2: self.loader2.release()
        self.loader2=VideoLoader(source=source)
        self.detect2=MotionDetector()
        if not self.loader2.is_open():
            self._log(f"Cam 2 failed to open: {name}"); return
        # Start its own dedicated thread
        self._cam2_running=True
        threading.Thread(target=self._cam2_loop,daemon=True).start()
        self.root.after(0,lambda: self.s_cam2.config(text="Active"))
        self._log(f"Cam 2 active: {name}")

    def _cam2_loop(self):
        """Reads and processes cam2 independently — never blocks cam1."""
        while self._cam2_running:
            if not self.loader2 or not self.loader2.is_open():
                break
            ret,frame=self.loader2.read_frame()
            if not ret:
                # Loop video back to start
                self.loader2.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                continue
            frame=cv2.resize(frame,(640,360))
            raw=frame.copy()  # raw for stitch
            boxes,_,_=self.detect2.detect(frame)
            boxes=merge_boxes(boxes,margin=6)
            for x1,y1,x2,y2 in boxes:
                lbl=classify(x1,y1,x2,y2,False,False)
                c=CLR[lbl]
                cv2.rectangle(frame,(x1,y1),(x2,y2),c,2)
                cv2.rectangle(frame,(x1,max(0,y1-18)),(x2,y1),c,-1)
                cv2.putText(frame,lbl,(x1+2,max(1,y1-3)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
            # Store with lock so stitch can safely read
            with self._cam2_lock:
                self._cam2_frame=frame.copy()
                self._cam2_live =raw.copy()
            time.sleep(0.04)

    # ──────────────────────────────────────────────────────────────
    #  STITCH VIEW  — uses stored frames, never reads cap directly
    # ──────────────────────────────────────────────────────────────
    def stitch_view(self):
        with self._cam2_lock:
            cam2_ready = self._cam2_frame is not None
        if not cam2_ready:
            self._log("Start Cam 2 first, then click Stitch View!")
            return
        # If already open just bring it to front
        if hasattr(self,"_sw") and self._sw.winfo_exists():
            self._sw.lift(); return
        self._sw=tk.Toplevel(self.root)
        self._sw.title("Stitched View — CAM-01 + CAM-02")
        self._sw.configure(bg=BG_MAIN)
        tk.Label(self._sw,text="STITCHED CAMERA VIEW",
                 bg=BG_MAIN,fg=TEXT_DARK,
                 font=("Arial",10,"bold")).pack(pady=4)
        self._sw_lbl=tk.Label(self._sw,bg="#111")
        self._sw_lbl.pack(padx=8,pady=4)
        self._sw_lbl.imgtk=None
        self._stitch_update()

    def _stitch_update(self):
        if not hasattr(self,"_sw") or not self._sw.winfo_exists(): return
        try:
            # Cam1 — use stored live frame (no cap.read())
            f1=self._cam1_live
            with self._cam2_lock:
                f2=self._cam2_frame.copy() if self._cam2_frame is not None else None
            if f1 is not None and f2 is not None:
                f1r=cv2.resize(f1,(580,326))
                f2r=cv2.resize(f2,(580,326))
                # Side-by-side with divider
                div=np.full((326,6,3),200,dtype=np.uint8)
                stitched=np.hstack([f1r,div,f2r])
                # Header labels
                cv2.rectangle(stitched,(0,0),(580,22),(40,50,30),-1)
                cv2.rectangle(stitched,(586,0),(1166,22),(40,50,30),-1)
                cv2.putText(stitched,"CAM-01",(6,15),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,200,170),1)
                cv2.putText(stitched,"CAM-02",(592,15),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,200,170),1)
                rgb=cv2.cvtColor(stitched,cv2.COLOR_BGR2RGB)
                imgtk=ImageTk.PhotoImage(image=Image.fromarray(rgb))
                self._sw_lbl.imgtk=imgtk
                self._sw_lbl.config(image=imgtk)
        except Exception:
            pass
        self._sw.after(50,self._stitch_update)

    # ──────────────────────────────────────────────────────────────
    #  PLAYBACK CONTROLS
    # ──────────────────────────────────────────────────────────────
    def start(self):
        if self.running: return
        if self.loader is None:
            self.loader=VideoLoader(source=self.source)
        self.running=True; self.paused=False
        self.lbl_status.config(text="LIVE",fg=G)
        self.s_status.config(text="Running")
        self._log("System started")
        threading.Thread(target=self._loop,daemon=True).start()

    def pause(self):
        self.paused=not self.paused
        if self.paused:
            self.lbl_status.config(text="PAUSED",fg=A)
            self._log("Paused")
        else:
            self.lbl_status.config(text="LIVE",fg=G)
            self._log("Resumed")

    def stop(self):
        self.running=False
        self._cam2_running=False
        self.saver.stop_recording()
        for ld in [self.loader,self.loader2]:
            if ld: ld.release()
        self.loader=self.loader2=None
        self._cam2_frame=self._cam2_live=self._cam1_live=None
        self.root.after(0,lambda: self.s_cam2.config(text="Off"))
        self.lbl_status.config(text="OFFLINE",fg=R)
        self.s_status.config(text="Idle")
        self._log("System stopped")

    def export_clip(self):
        if not os.path.exists("saved_clips"):
            self._log("No clips folder!"); return
        clips=os.listdir("saved_clips")
        if not clips: self._log("No clips to export!"); return
        latest=sorted(clips)[-1]
        dst=filedialog.asksaveasfilename(
            defaultextension=".avi",initialfile=latest,
            filetypes=[("AVI files","*.avi"),("MP4 files","*.mp4")])
        if dst:
            import shutil
            shutil.copy(f"saved_clips/{latest}",dst)
            self._log(f"Exported: {latest}")

    # ──────────────────────────────────────────────────────────────
    #  DRAW BOXES + TRAILS
    # ──────────────────────────────────────────────────────────────
    def _draw(self,frame,boxes,tracks):
        hn=vn=mn=0
        for x1,y1,x2,y2 in boxes:
            lbl=classify(x1,y1,x2,y2,self.webcam,self.outdoor)
            c=CLR[lbl]
            if lbl=="HUMAN": hn+=1
            elif lbl=="VEHICLE": vn+=1
            else: mn+=1
            cv2.rectangle(frame,(x1,y1),(x2,y2),c,2)
            cv2.rectangle(frame,(x1,max(0,y1-18)),(x2,y1),c,-1)
            cv2.putText(frame,lbl,(x1+3,max(2,y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
        if self.do_track.get() and tracks:
            for tid,t in tracks.items():
                trail=t["trail"]; tc=CLR.get(t["label"],(120,100,80))
                if self.do_trails.get() and len(trail)>1:
                    for k in range(1,len(trail)):
                        alpha=k/len(trail)
                        c2=tuple(int(v*alpha) for v in tc)
                        cv2.line(frame,trail[k-1],trail[k],c2,2)
                if trail:
                    cx,cy=trail[-1]
                    cv2.putText(frame,f"#{tid}",(cx+3,cy-3),
                                cv2.FONT_HERSHEY_SIMPLEX,0.3,(80,60,40),1)
        return frame,hn,vn,mn

    # ──────────────────────────────────────────────────────────────
    #  MAIN PROCESSING LOOP  (cam1 only — cam2 is on its own thread)
    # ──────────────────────────────────────────────────────────────
    def _loop(self):
        while self.running:
            if self.paused: time.sleep(0.05); continue
            if not self.loader or not self.loader.is_open(): break
            ret,frame=self.loader.read_frame()
            if not ret: break

            frame=cv2.resize(frame,(860,480))
            if self.do_undistort.get(): frame=undistort(frame)
            if self.do_compress.get():  frame=compress(frame,self.quality.get())

            orig=frame.copy()
            # Store raw frame for stitch window (no cap conflict)
            self._cam1_live=orig.copy()

            boxes,mask,heat=self.detector.detect(frame)
            boxes=merge_boxes(boxes)

            labelled=[(x1,y1,x2,y2,
                       classify(x1,y1,x2,y2,self.webcam,self.outdoor))
                      for x1,y1,x2,y2 in boxes]
            tracks=self.tracker.update(labelled) if self.do_track.get() else {}

            det=frame.copy()
            det,hn,vn,mn=self._draw(det,boxes,tracks)

            human_now=hn>0; vehicle_now=vn>0; motion_now=len(boxes)>0

            if self.saver.is_recording:
                cv2.circle(det,(det.shape[1]-28,18),8,(154,58,58),-1)
                cv2.putText(det,"REC",(det.shape[1]-60,50),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,(154,58,58),1)

            # Save annotated clip (boxes visible in recording)
            self.saver.update(det,human_now)

            # Convert panels — store in self.img to prevent GC flicker
            i0=to_imgtk(heat); i1=to_imgtk(orig)
            i2=to_imgtk(det);  i3=mask_imgtk(mask)

            def update_panels(h=i0,l=i1,d=i2,m=i3):
                self.img=[h,l,d,m]  # strong refs
                self.p_heat.config(image=h)
                self.p_live.config(image=l)
                self.p_det.config(image=d)
                self.p_mask.config(image=m)

            self.root.after(0,update_panels)
            self.root.after(0,lambda a=hn,b=vn,c=mn:(
                self.c_human.config(text=str(a)),
                self.c_vehicle.config(text=str(b)),
                self.c_motion.config(text=str(c))))

            now=time.time()
            if human_now:
                if not self.prev_human:
                    self.n_human+=1
                    self.root.after(0,lambda n=self.n_human:
                        self.s_human.config(text=str(n)))
                self.prev_human=True; self.prev_vehicle=self.prev_motion=False
                self.root.after(0,lambda:self.alert.config(
                    text="  !! HUMAN DETECTED - UNAUTHORISED ACCESS !!",
                    bg=R,fg="#fff"))
                if now-self.last_log>3:
                    self._log("Human detected!"); self.last_log=now
            elif vehicle_now:
                if not self.prev_vehicle:
                    self.n_vehicle+=1
                    self.root.after(0,lambda n=self.n_vehicle:
                        self.s_vehicle.config(text=str(n)))
                self.prev_vehicle=True; self.prev_human=self.prev_motion=False
                self.root.after(0,lambda:self.alert.config(
                    text="  VEHICLE DETECTED",bg=A,fg="#fff"))
                if now-self.last_log>3:
                    self._log("Vehicle detected!"); self.last_log=now
            elif motion_now:
                if not self.prev_motion:
                    self.n_motion+=1
                    self.root.after(0,lambda n=self.n_motion:
                        self.s_motion.config(text=str(n)))
                self.prev_motion=True; self.prev_human=self.prev_vehicle=False
                self.root.after(0,lambda:self.alert.config(
                    text="  MOTION DETECTED",bg=A,fg="#fff"))
            else:
                self.prev_human=self.prev_vehicle=self.prev_motion=False
                self.root.after(0,lambda:self.alert.config(
                    text="  Monitoring...",bg=G,fg="#fff"))

            self.root.after(0,self._refresh_clips)
            time.sleep(0.05)

        self.running=False
        self._log("Playback finished")


if __name__=="__main__":
    root=tk.Tk()
    app=Dashboard(root)
    root.mainloop()