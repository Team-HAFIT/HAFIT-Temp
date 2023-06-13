import React, { useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
import '@tensorflow/tfjs-backend-webgl';
import './style.css';

const Movenet = () => {
    let detector;
    let detectorConfig;
    let poses;
    let skeleton = true;
    const confidence_threshold = 0.5;
    let video, ctx, canvas;
    let hurrycheckpoint;
    let hurrycheck;
    let squatCount = 0;
    let squatStarted = false;
    let squatFinished = false;
    const kneeAngleThreshold = 130;
    const redhurryAngleThreshold = 50;
    const orangehurryAngleThreshold = 30;
    let currentSet = 1;
    let repsPerSet = 3;
    let totalSets = 3;
    let restStarted = false;
    let restTime = 10;

    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    const edges = {                               // keypoint의 선을 잇는다.
        '5,7': 'm',
        '5,17': 'm',
        '6,17': 'm',
        '7,9': 'm',
        '6,8': 'm',
        '8,10': 'm',
        '11,13': 'm',
        '13,15': 'm',
        '12,14': 'm',
        '14,16': 'm',
        '17,18': 'm',
        '18,19': 'm',
        '19,20': 'm',
        '12,20': 'm',
        '11,20': 'm',
      };
      
      const hurry_check_edges={
        '5,7': 'm',
        '5,17': 'm',
        '6,17': 'm',
        '7,9': 'm',
        '6,8': 'm',
        '8,10': 'm',
        '11,13': 'm',
        '13,15': 'm',
        '12,14': 'm',
        '14,16': 'm',
        '12,20': 'm',
        '11,20': 'm',
      }
      
      const hurry_error_edges= {
        '17,18': 'm',
        '18,19': 'm',
        '19,20': 'm'
      }

    useEffect(() => {
        init();
    }, []);

    const init = async () => {
        tf.setBackend('webgpu');
        detectorConfig = {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
            enableSmoothing: true,
            enableSegmentation: false,
        };
        detector = await poseDetection.createDetector(
            poseDetection.SupportedModels.MoveNet,
            detectorConfig
        );
        setup();
        draw();
    }

    const setup = async () => {
        canvas = canvasRef.current;
        ctx = canvas.getContext('2d');
        video = videoRef.current;

        const camera = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
        });
        video.srcObject = camera;

        // Add an event listener for the 'loadeddata' event
        video.addEventListener('loadeddata', () => {
            video.play();
        });

        const onLoadedMetadata = () => {
            // Start detecting poses once the video dimensions are set.
            getPoses();
        };
        video.addEventListener('loadedmetadata', onLoadedMetadata);
    }


    const getPoses = async () => {
        poses = await detector.estimatePoses(video);
        setTimeout(getPoses, 0);
        if (poses && poses.length > 0) {
            const leftShoulder = poses[0].keypoints[5];
            const rightShoulder = poses[0].keypoints[6];
            const leftHip = poses[0].keypoints[11];
            const rightHip = poses[0].keypoints[12];
            const midShoulderX = (leftShoulder.x + rightShoulder.x) / 2;
            const midShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
            const midHipX = (leftHip.x + rightHip.x) / 2;
            const midHipY = (leftHip.y + rightHip.y) / 2;
            const middleShoulder = { x: midShoulderX, y: midShoulderY, score: Math.min(leftShoulder.score, rightShoulder.score) };
            poses[0].keypoints[17] = { x: midShoulderX, y: midShoulderY, score: Math.min(leftShoulder.score, rightShoulder.score) };
            const midhip = { x: midHipX, y: midHipY, score: Math.min(leftHip.score, rightHip.score) };
            poses[0].keypoints[20] = { x: midHipX, y: midHipY, score: Math.min(leftHip.score, rightHip.score) };
            const x1 = middleShoulder.x;
            const y1 = middleShoulder.y;
            const x2 = poses[0].keypoints[20].x;
            const y2 = poses[0].keypoints[20].y;
            const x1_3 = (2 * x1 + x2) / 3;
            const y1_3 = (2 * y1 + y2) / 3;
            const x2_3 = (x1 + 2 * x2) / 3;
            const y2_3 = (y1 + 2 * y2) / 3;
            poses[0].keypoints[18] = { x: x1_3, y: y1_3, score: Math.min(leftShoulder.score, rightShoulder.score) };
            poses[0].keypoints[19] = { x: x2_3, y: y2_3, score: Math.min(leftHip.score, rightHip.score) };
            hurrycheckpoint = { x: midHipX, y: midShoulderY, score: Math.min(midhip.score, middleShoulder.score) };
            hurrycheck = calculateAngle(middleShoulder, midhip, hurrycheckpoint);
        }
    }

    const draw = () => {
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        
        ctx.drawImage(video, 0, 0, video.width, video.height);
        
        drawKeypoints();
        if (skeleton) {
            drawSkeleton();
        }
        ctx.restore();
        ctx.fillStyle = 'blue';
        ctx.strokeStyle = 'blue';
        ctx.lineWidth = 2;
        ctx.font = '30px Arial';
        ctx.save();
        if (poses && poses.length > 0) {
            countSquats();
            ctx.scale(-1, 1); // 텍스트 좌우 반전을 위해 캔버스 좌우 반전
            ctx.fillText('Pose detected', -canvas.width + 10, 50);
            ctx.strokeText('Pose detected', -canvas.width + 10, 50);
            ctx.fillText(`스쿼트 : ${squatCount}`, -canvas.width + 10, 120);
            ctx.strokeText(`스쿼트 : ${squatCount}`, -canvas.width + 10, 120);
            ctx.fillText(`세트: ${currentSet} / ${totalSets}`, -canvas.width + 10, 190);
            ctx.strokeText(`세트: ${currentSet} / ${totalSets}`, -canvas.width + 10, 190);
        } else {
            ctx.scale(-1, 1); // 텍스트 좌우 반전을 위해 캔버스 좌우 반전
            ctx.fillText('No pose detected', 10, 50);
            ctx.strokeText('No pose detected', 10, 50);
        }
        ctx.restore();
        window.requestAnimationFrame(draw);
      }

      const drawKeypoints = () => {
        var count = 0;
        if (poses && poses.length > 0) {
          const canvasWidth = canvas.width;
          const canvasHeight = canvas.height;
          const originalWidth = video.videoWidth;
          const originalHeight = video.videoHeight;
          const widthRatio = canvasWidth / originalWidth;
          const heightRatio = canvasHeight / originalHeight;
          ctx.save(); // 현재의 그래픽 상태 저장
      
          ctx.font = '10px Arial'; // 글자 크기 설정
      
          for (let kp of poses[0].keypoints) {
            const { x, y, score } = kp;
            const adjustedX = x * widthRatio;
            const adjustedY = y * heightRatio;
            if (score > confidence_threshold) {
              count = count + 1;
              ctx.fillStyle = 'white';
              ctx.strokeStyle = 'black';
              ctx.lineWidth = 4;
              ctx.beginPath();
              ctx.arc(adjustedX, adjustedY, 8, 0, 2 * Math.PI);
              ctx.fill();
              ctx.stroke();
      
              
            }
          }
      
          ctx.restore(); // 그래픽 상태를 원래대로 복원
        }
      }

    const drawSkeleton = () => {
        if (poses && poses.length > 0) {
            const canvasWidth = canvas.width;
            const canvasHeight = canvas.height;
            const originalWidth = video.videoWidth;
            const originalHeight = video.videoHeight;
            const widthRatio = canvasWidth / originalWidth;
            const heightRatio = canvasHeight / originalHeight;
            if (orangehurryAngleThreshold > hurrycheck && hurrycheck < redhurryAngleThreshold) {
                for (const [key, value] of Object.entries(hurry_check_edges)) {
                    const p = key.split(",");
                    const p1 = parseInt(p[0]);
                    const p2 = parseInt(p[1]);
                    const kp1 = poses[0].keypoints[p1];
                    const kp2 = poses[0].keypoints[p2];
                    const x1 = kp1.x * widthRatio;
                    const y1 = kp1.y * heightRatio;
                    const c1 = kp1.score;
                    const x2 = kp2.x * widthRatio;
                    const y2 = kp2.y * heightRatio;
                    const c2 = kp2.score;
                    if (c1 > confidence_threshold && c2 > confidence_threshold) {
                        ctx.strokeStyle = 'rgb(0, 255, 0)';
                        ctx.lineWidth = 6; // 라인의 두께를 변경합니다.
                        ctx.beginPath();
                        ctx.moveTo(x1, y1);
                        ctx.lineTo(x2, y2);
                        ctx.stroke();
                    }
                }
                for (const [key, value] of Object.entries(hurry_error_edges)) {
                    const p = key.split(",");
                    const p1 = parseInt(p[0]);
                    const p2 = parseInt(p[1]);
                    const kp1 = poses[0].keypoints[p1];
                    const kp2 = poses[0].keypoints[p2];
                    const x1 = kp1.x * widthRatio;
                    const y1 = kp1.y * heightRatio;
                    const c1 = kp1.score;
                    const x2 = kp2.x * widthRatio;
                    const y2 = kp2.y * heightRatio;
                    const c2 = kp2.score;
                    if (c1 > confidence_threshold && c2 > confidence_threshold) {
                        ctx.strokeStyle = 'rgb(255, 0, 0)';
                        ctx.lineWidth = 6; // 라인의 두께를 변경합니다.
                        ctx.beginPath();
                        ctx.moveTo(x1, y1);
                        ctx.lineTo(x2, y2);
                        ctx.stroke();
                    }
                }
            }else if (hurrycheck > redhurryAngleThreshold){
                for (const [key, value] of Object.entries(hurry_check_edges)) {
                    const p = key.split(",");
                    const p1 = parseInt(p[0]);
                    const p2 = parseInt(p[1]);
                    const kp1 = poses[0].keypoints[p1];
                    const kp2 = poses[0].keypoints[p2];
                    const x1 = kp1.x * widthRatio;
                    const y1 = kp1.y * heightRatio;
                    const c1 = kp1.score;
                    const x2 = kp2.x * widthRatio;
                    const y2 = kp2.y * heightRatio;
                    const c2 = kp2.score;
                    if (c1 > confidence_threshold && c2 > confidence_threshold) {
                        ctx.strokeStyle = 'rgb(0, 255, 0)';
                        ctx.lineWidth = 6; // 라인의 두께를 변경합니다.
                        ctx.beginPath();
                        ctx.moveTo(x1, y1);
                        ctx.lineTo(x2, y2);
                        ctx.stroke();
                    }
                }
                for (const [key, value] of Object.entries(hurry_error_edges)) {
                    const p = key.split(",");
                    const p1 = parseInt(p[0]);
                    const p2 = parseInt(p[1]);
                    const kp1 = poses[0].keypoints[p1];
                    const kp2 = poses[0].keypoints[p2];
                    const x1 = kp1.x * widthRatio;
                    const y1 = kp1.y * heightRatio;
                    const c1 = kp1.score;
                    const x2 = kp2.x * widthRatio;
                    const y2 = kp2.y * heightRatio;
                    const c2 = kp2.score;
                    if (c1 > confidence_threshold && c2 > confidence_threshold) {
                        ctx.strokeStyle = 'rgb(0, 0, 0)';
                        ctx.lineWidth = 6; // 라인의 두께를 변경합니다.
                        ctx.beginPath();
                        ctx.moveTo(x1, y1);
                        ctx.lineTo(x2, y2);
                        ctx.stroke();
                    }
                }
            }else {
                for (const [key, value] of Object.entries(edges)) {
                    const p = key.split(",");
                    const p1 = parseInt(p[0]);
                    const p2 = parseInt(p[1]);
                    const kp1 = poses[0].keypoints[p1];
                    const kp2 = poses[0].keypoints[p2];
                    const x1 = kp1.x * widthRatio;
                    const y1 = kp1.y * heightRatio;
                    const c1 = kp1.score;
                    const x2 = kp2.x * widthRatio;
                    const y2 = kp2.y * heightRatio;
                    const c2 = kp2.score;
                    if (c1 > confidence_threshold && c2 > confidence_threshold) {
                        ctx.strokeStyle = 'rgb(0, 255, 0)';
                        ctx.lineWidth = 6; // 라인의 두께를 변경합니다.
                        ctx.beginPath();
                        ctx.moveTo(x1, y1);
                        ctx.lineTo(x2, y2);
                        ctx.stroke();
                    }
                }
            }
        }
    }

    const countSquats = () => {
        const kneeKeypointsConfident = poses[0].keypoints.slice(10, 16).every(kp => kp.score > confidence_threshold);
        if (kneeKeypointsConfident) {
            const leftHip = poses[0].keypoints[11];
            const rightHip = poses[0].keypoints[12];
            const leftKnee = poses[0].keypoints[13];
            const rightKnee = poses[0].keypoints[14];
            const leftAnkle = poses[0].keypoints[15];
            const rightAnkle = poses[0].keypoints[16];
            const leftKneeAngle = calculateAngle(leftHip, leftKnee, leftAnkle);
            const rightKneeAngle = calculateAngle(rightHip, rightKnee, rightAnkle);
            if (!squatStarted && leftKneeAngle <= kneeAngleThreshold && rightKneeAngle <= kneeAngleThreshold) {
                squatStarted = true;
                squatFinished = false;
            }
            if (squatStarted && leftKneeAngle > kneeAngleThreshold && rightKneeAngle > kneeAngleThreshold) {
                squatCount++;
                squatStarted = false;
                squatFinished = true;
            }
            if (squatCount > 0 && squatFinished) {
                console.log(`스쿼트 ${squatCount}회 완료!`);
                squatFinished = false;
            }
        }
    }

    const calculateAngle = (p1, p2, p3) => {
        let dx1 = p1.x - p2.x;
        let dy1 = p1.y - p2.y;
        let dx2 = p3.x - p2.x;
        let dy2 = p3.y - p2.y;
        let dot = dx1 * dx2 + dy1 * dy2;
        let mag1 = Math.sqrt(dx1 * dx1 + dy1 * dy1);
        let mag2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);
        let angle = Math.acos(dot / (mag1 * mag2));
        return angle * (180.0 / Math.PI);
    }

    return (
        <div id='container' >
            <video id="webcam" ref={videoRef} width="900" height="700" autoPlay></video>
            <canvas id="movenet_Canvas" ref={canvasRef} width="900" height="700"></canvas>
        </div>
    );
}

export default Movenet;