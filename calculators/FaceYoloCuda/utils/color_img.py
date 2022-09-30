import pyrealsense2 as rs2

class ColorImg:
    def __init__(self,colorframe,rs_intrinsics):
        self.colorframe = colorframe
        self.intrinsics = rs_intrinsics

