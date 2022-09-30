import pyrealsense2 as rs
import cv2
import numpy as np
import math
import os

if __name__=='__main__':
    ctx=rs.context()
    devices=ctx.query_devices()
    serials=[]
    pipelines=[]
    colorizers={}
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rgb_intrinsics=[]
    depth_intrinsics = []
    cms=[]
    coeffs=[]

    align=rs.align(rs.stream.color)


    for item in devices:
        serial=item.get_info(rs.camera_info.serial_number)
        serials.append(serial)

    cnt=0
    for serial in serials:
        pipeline=rs.pipeline(ctx)
        cfg=rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        if(cnt==0):
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = cfg.resolve(pipeline)
        c=pipeline.start(cfg)
        pipelines.append(pipeline)

        if cnt==0:
            depth_sensor=profile.get_device().first_depth_sensor()
            depth_sensor.set_option(rs.option.visual_preset, 5)
            depth_profile = c.get_stream(rs.stream.depth)
            depth_intrinsic = depth_profile.as_video_stream_profile().get_intrinsics()
            depth_intrinsics.append(depth_intrinsic)

        rgb_profile = c.get_stream(rs.stream.color)
        rgb_intrinsic = rgb_profile.as_video_stream_profile().get_intrinsics()
        rgb_intrinsics.append(rgb_intrinsic)
        cm=np.array([[rgb_intrinsic.fx,0,rgb_intrinsic.ppx],[0,rgb_intrinsic.fy,rgb_intrinsic.ppy],[0,0,1]])
        coeff=np.array(rgb_intrinsic.coeffs)
        cms.append(cm)
        coeffs.append(coeff)
        cnt+=1

    pre = np.array([0, 0, 0]).reshape(3, 1)
    stablecnt = 0
    while True:
        framesets=[]
        objectpoints=[]
        objpoints=np.zeros((4*6,3), np.float32)
        for i in range(0, 6):
            for j in range(0, 4):
                objpoints[4*i+j, :2] = [38.30 * j, 38.30 * i]
        imagepoints0=[]
        imagepoints1=[]
        objectpoints.append(objpoints)

        #rgb+depth
        objectpoints2=[]
        depths=[]

        cnt=0
        for pipeline in pipelines:
            frameset=pipeline.wait_for_frames()
            if cnt==0:
                aligned=align.process(frameset)
                framesets.append(aligned)
            else:
                framesets.append(frameset)

        color0 = framesets[0].get_color_frame()
        depth0 = framesets[0].get_depth_frame()

        color0 = np.asanyarray(color0.get_data())
        depth0_ = np.asanyarray(depth0.get_data())

        gray0 = cv2.cvtColor(color0, cv2.COLOR_BGR2GRAY)

        ret0, corners0 = cv2.findChessboardCorners(gray0, (4, 6), None)


        if ret0 == True:
            imagepoints0.append(corners0)
            corners02 = cv2.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            cv2.drawChessboardCorners(color0, (4, 6), corners02, ret0)
            for i in range(0,len(corners02)):
                p=(int(corners02[i][0][0]),int(corners02[i][0][1]))
                dph=depth0.get_distance(p[0],p[1])
                obj2 = rs.rs2_deproject_pixel_to_point(rgb_intrinsics[0], p, dph)
                objectpoints2.append(obj2)
                depths.append(dph)
                cv2.putText(color0,str(i+1),p,cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2,2)

            objectpoints2 = np.array(objectpoints2)

            imgpoints0 = corners0.reshape(-1, 2)
            _, r0, t0 = cv2.solvePnP(objectPoints=objectpoints[0], imagePoints=imgpoints0, cameraMatrix=cms[0],
                                     distCoeffs=coeffs[0], flags=cv2.SOLVEPNP_ITERATIVE)

            R0, _ = cv2.Rodrigues(r0)
            R0 = R0.T
            t0 = -np.matmul(R0, t0)
            cv2.putText(color0, 't=[{},{},{}]'.format(t0[0], t0[1], t0[2]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, 2)

            projs = []
            for i in range(0, 24):
                proj = np.matmul(R0, objectpoints2[i].reshape(3, 1) * 1000) + t0
                proj_ = np.matmul(cms[0], proj / proj[2]).squeeze(1)[0:2]
                projs.append(proj_)
                cv2.circle(img=color0, center=(int(proj_[0]), int(proj_[1])), radius=6, color=(255, 0, 255))
            projs = np.array(projs)
            corners1 = corners0.squeeze(1).reshape(-1, 2)
            diff = np.sqrt(np.sum((projs - corners0) * (projs - corners0), axis=1))
            cv2.putText(color0, 'max_error={:.2}, min_error={:.2}'.format(np.max(diff), np.min(diff)), (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, 2)


            dif=(t0-pre).sum()
            print(dif)

            if dif<20:
                stablecnt+=1
            else:
                stablecnt=0

            pre=t0

            if(stablecnt>50):
                f=open('0toscreen.txt','w',encoding='utf-8')
                for i in range(0,R0.shape[0]):
                    for j in range(0,R0.shape[1]):
                        f.write(str(R0[i][j])+' ')
                    f.write('\n')

                #f.write(str(t0[0][0])+' '+str(t_[1][0])+' '+str(t_[2][0])+'\n')
                f.close()
                exit(0)


        cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('img', color0)
        cv2.waitKey(1)






