import time
from ScreenViewer import ScreenViewer

if __name__ == "__main__":
    sv = ScreenViewer()
    sv.GetHWND('Unity 2018.4.14f1 Personal - PowerfulMagnet (modify).unity - UnityMagnets-master - PC, Mac & Linux Standalone <DX11>')
    sv.Start()
    time.sleep(1)
    sv.Stop()