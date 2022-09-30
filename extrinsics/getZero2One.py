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
        objpoints=np.zeros((6*4,3), np.float32)
        for i in range(0, 4):
            for j in range(0, 6):
                objpoints[6*i+j, :2] = [38.30 * j, 38.30 * i]
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
        color1 = framesets[1].get_color_frame()

        color0 = np.asanyarray(color0.get_data())
        color1 = np.asanyarray(color1.get_data())

        depth0_ = np.asanyarray(depth0.get_data())

        gray0 = cv2.cvtColor(color0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)

        ret0, corners0 = cv2.findChessboardCorners(gray0, (6, 4), None)
        ret1, corners1 = cv2.findChessboardCorners(gray1, (6, 4), None)

        if ret0 == True:
            imagepoints0.append(corners0)
            corners02 = cv2.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            cv2.drawChessboardCorners(color0, (6, 4), corners02, ret0)
            for i in range(0,len(corners02)):
                p=(int(corners02[i][0][0]),int(corners02[i][0][1]))
                dph=depth0.get_distance(p[0],p[1])
                obj2 = rs.rs2_deproject_pixel_to_point(rgb_intrinsics[0], p, dph)
                objectpoints2.append(obj2)
                depths.append(dph)
                cv2.putText(color0,str(i+1),p,cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2,2)

            objectpoints2 = np.array(objectpoints2)


        if ret1 == True:
            imagepoints1.append(corners1)
            corners12 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            # Draw and display the corners
            cv2.drawChessboardCorners(color1, (6, 4), corners12, ret1)
            for i in range(0,len(corners12)):
                p=(int(corners12[i][0][0]),int(corners12[i][0][1]))
                cv2.putText(color1,str(i+1),p,cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2,2)



        if(ret0 and ret1):
            imgpoints0 = corners0.reshape(-1, 2)
            imgpoints1 = corners1.reshape(-1, 2)

            _,r0, t0 = cv2.solvePnP(objectPoints=objectpoints[0], imagePoints=imgpoints0, cameraMatrix=cms[0], distCoeffs=coeffs[0], flags=cv2.SOLVEPNP_ITERATIVE)
            _,r1, t1 = cv2.solvePnP(objectPoints=objectpoints[0], imagePoints=imgpoints1, cameraMatrix=cms[1], distCoeffs=coeffs[1], flags=cv2.SOLVEPNP_ITERATIVE)

            R0,_ = cv2.Rodrigues(r0)
            R1,_ = cv2.Rodrigues(r1)

            R_ = np.matmul(R1, R0.T)
            t_ = -np.matmul(np.matmul(R1, R0.T), t0) + t1

            R10 = np.matmul(R0, R1.T)
            t10 = -np.matmul(np.matmul(R0, R1.T), t1) + t0

            cv2.putText(color0, 't=[{},{},{}]'.format(t_[0], t_[1], t_[2]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 2, 2)



            _, r01, t01 = cv2.solvePnP(objectPoints=objectpoints2, imagePoints=imgpoints1, cameraMatrix=cms[1],distCoeffs=coeffs[1], flags=cv2.SOLVEPNP_ITERATIVE)
            R01, _ = cv2.Rodrigues(r01)


            projs=[]
            for i in range(0,24):
                proj = np.matmul(R_, objectpoints2[i].reshape(3, 1) * 1000) + t_
                proj_=np.matmul(cms[1],proj/proj[2]).squeeze(1)[0:2]
                projs.append(proj_)
                cv2.circle(img=color1,center=(int(proj_[0]),int(proj_[1])),radius=6,color=(255,0,255))
            projs=np.array(projs)
            corners1=corners1.squeeze(1).reshape(-1,2)
            diff=np.sqrt(np.sum((projs-corners1)*(projs-corners1),axis=1))


            cv2.putText(color0, 'max_error={:.2}, min_error={:.2}'.format(np.max(diff),np.min(diff)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, 2)


            dif=(t_-pre).sum()
            print(dif)

            if dif<20:
                stablecnt+=1
            else:
                stablecnt=0

            pre=t_

            if(stablecnt>50):
                f=open('0to1.txt','w',encoding='utf-8')
                for i in range(0,R_.shape[0]):
                    for j in range(0,R_.shape[1]):
                        f.write(str(R_[i][j])+' ')
                    f.write('\n')

                f.write(str(t_[0][0])+' '+str(t_[1][0])+' '+str(t_[2][0])+'\n')
                f.close()

                f2 = open('1to0.txt', 'w', encoding='utf-8')
                for i in range(0, R10.shape[0]):
                    for j in range(0, R10.shape[1]):
                        f2.write(str(R10[i][j]) + ' ')
                    f2.write('\n')

                f2.write(str(t10[0][0]) + ' ' + str(t10[1][0]) + ' ' + str(t10[2][0]) + '\n')
                f2.close()

                exit(0)


        merge=cv2.hconcat([color0,color1])
        cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('img', merge)
        cv2.waitKey(1)






