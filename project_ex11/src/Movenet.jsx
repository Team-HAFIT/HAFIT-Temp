import React, { useEffect, useRef, useState } from 'react';
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
  
  let squatStarted = false;
  let squatFinished = false;
  const kneeAngleThreshold = 130;
  const orangehurryAngleThreshold = 30;
  const redhurryAngleThreshold = 50;
  let currentSet = 1;
  let totalSets = 3;

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
 
  const [timer, setTimer] = useState(0);
  const timerRef = useRef(null);
  const [isPoseDetected, setIsPoseDetected] = useState(false);
  const startTimeRef = useRef(null);
  const [squatCount, setSquatCount] = useState(0);

  const edges = {
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

  const hurry_check_edges = {
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
  };

  const hurry_error_edges = {
    '17,18': 'm',
    '18,19': 'm',
    '19,20': 'm',
  };


  useEffect(() => {
    init();
    startTimer();
  }, []);

  useEffect(() => {
    if (isPoseDetected) {
      startTimer();
    } else {
      stopTimer();
    }
  }, [isPoseDetected]);

  const startTimer = () => {
    if (!timerRef.current) {
      startTimeRef.current = Date.now() - (timer * 1000); // Subtract elapsed time from current time
      timerRef.current = setInterval(() => {
        const elapsedSeconds = Math.floor((Date.now() - startTimeRef.current) / 1000);
        setTimer(elapsedSeconds);
      }, 1000);
    }
  };

  const stopTimer = () => {
    clearInterval(timerRef.current);
    timerRef.current = null;
  };

  
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
  };

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
  };

  const getPoses = async () => {
    poses = await detector.estimatePoses(video);
    setTimeout(getPoses, 0);
 
    if (poses && poses.length > 0) {
      setIsPoseDetected(true);
      const leftShoulder = poses[0].keypoints[5];
      const rightShoulder = poses[0].keypoints[6];
      const leftHip = poses[0].keypoints[11];
      const rightHip = poses[0].keypoints[12];
      const midShoulderX = (leftShoulder.x + rightShoulder.x) / 2;
      const midShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
      const midHipX = (leftHip.x + rightHip.x) / 2;
      const midHipY = (leftHip.y + rightHip.y) / 2;
      const middleShoulder = {
        x: midShoulderX,
        y: midShoulderY,
        score: Math.min(leftShoulder.score, rightShoulder.score),
      };
      poses[0].keypoints[17] = {
        x: midShoulderX,
        y: midShoulderY,
        score: Math.min(leftShoulder.score, rightShoulder.score),
      };
      const midhip = {
        x: midHipX,
        y: midHipY,
        score: Math.min(leftHip.score, rightHip.score),
      };
      poses[0].keypoints[20] = {
        x: midHipX,
        y: midHipY,
        score: Math.min(leftHip.score, rightHip.score),
      };
      const x1 = middleShoulder.x;
      const y1 = middleShoulder.y;
      const x2 = poses[0].keypoints[20].x;
      const y2 = poses[0].keypoints[20].y;
      const x1_3 = (2 * x1 + x2) / 3;
      const y1_3 = (2 * y1 + y2) / 3;
      const x2_3 = (x1 + 2 * x2) / 3;
      const y2_3 = (y1 + 2 * y2) / 3;
      poses[0].keypoints[18] = {
        x: x1_3,
        y: y1_3,
        score: Math.min(leftShoulder.score, rightShoulder.score),
      };
      poses[0].keypoints[19] = {
        x: x2_3,
        y: y2_3,
        score: Math.min(leftHip.score, rightHip.score),
      };
      hurrycheckpoint = {
        x: midHipX,
        y: midShoulderY,
        score: Math.min(midhip.score, middleShoulder.score),
      };
      hurrycheck = calculateAngle(middleShoulder, midhip, hurrycheckpoint);
    } else{
      setIsPoseDetected(false);
    }
    
  };

 


  const draw = () => {
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();

    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);

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
    } else {

    }
    ctx.restore();
    window.requestAnimationFrame(draw);
  };

  const drawShapes = (ctx) => {
    const centerX = 450; // 중심 X 좌표
    const centerY = 40; // 중심 Y 좌표
    const radius = 25; // 원의 반지름
    const gap = 20; // 원과 원 사이의 간격
    
    const originalStrokeStyle = ctx.strokeStyle;

    for (let i = 0; i < totalSets; i++) {
      const x = centerX + (radius * 2 + gap) * i;
      const y = centerY;
      
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
  
      if (i < currentSet) {
        ctx.fillStyle = 'green'; // 완료된 세트에 초록색으로 채우기
      } else {
        ctx.fillStyle = 'white';
      }
      ctx.strokeStyle = 'yellow'; // Set stroke color to yellow
      ctx.fill();
      ctx.stroke();
    }
    ctx.strokeStyle = originalStrokeStyle;
  };

  const drawKeypoints = () => {
    var count = 0;
    if (poses && poses.length > 0) {
     
      const canvasWidth = canvas.width;
      const canvasHeight = canvas.height;
      const originalWidth = video.videoWidth;
      const originalHeight = video.videoHeight;
      const widthRatio = canvasWidth / originalWidth;
      const heightRatio = canvasHeight / originalHeight;
      ctx.save();

      ctx.font = '10px Arial';

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
      ctx.restore();
    } else {
      
    }
  };



  const drawSkeleton = () => {
    if (poses && poses.length > 0) {
      const canvasWidth = canvas.width;
      const canvasHeight = canvas.height;
      const originalWidth = video.videoWidth;
      const originalHeight = video.videoHeight;
      const widthRatio = canvasWidth / originalWidth;
      const heightRatio = canvasHeight / originalHeight;

      if (hurrycheck > orangehurryAngleThreshold && hurrycheck <= redhurryAngleThreshold) {
        for (const [key, value] of Object.entries(hurry_check_edges)) {
          const p = key.split(',');
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
            ctx.strokeStyle = 'rgb(0, 255, 0)'; // 초록색
            ctx.lineWidth = 6;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
        for (const [key, value] of Object.entries(hurry_error_edges)) {
          const p = key.split(',');
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
            ctx.strokeStyle = 'rgb(255, 165, 0)'; // 주황색
            ctx.lineWidth = 6;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
      } else if (hurrycheck > redhurryAngleThreshold) {
        for (const [key, value] of Object.entries(hurry_check_edges)) {
          const p = key.split(',');
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
            ctx.strokeStyle = 'rgb(0, 255, 0)'; // 초록색
            ctx.lineWidth = 6;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
        for (const [key, value] of Object.entries(hurry_error_edges)) {
          const p = key.split(',');
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
            ctx.strokeStyle = 'rgb(255, 0, 0)'; // 빨간색
            ctx.lineWidth = 6;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
      } else {
        for (const [key, value] of Object.entries(edges)) {
          const p = key.split(',');
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
            ctx.strokeStyle = 'rgb(0, 255, 0)'; // 초록색
            ctx.lineWidth = 6;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
      }
    }
  };

  const countSquats = () => {
    const kneeKeypointsConfident = poses[0].keypoints
      .slice(10, 16)
      .every(kp => kp.score > confidence_threshold);
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
        setSquatCount(prevCount => prevCount + 1);
        squatStarted = false;
        squatFinished = true;
      }
      if (squatCount > 0 && squatFinished) {
        console.log(`스쿼트 ${squatCount}회 완료!`);
        squatFinished = false;
      }
    }
  };

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
  };

  return (
    <div id='container'>
      <div id = "state">
        <div id="shapeContainer">
        {Array.from({ length: totalSets }, (_, index) => (
          <div
            key={index}
            className={index < currentSet ? 'shape completed' : 'shape'}
            ></div>
          ))}
        </div>
        <div>타이머: {timer}초</div>
        <div>스쿼트 카운트: {squatCount}</div>
        <div>Pose 감지 상태: {isPoseDetected ? '감지됨' : '감지 안됨'}</div>
      </div>
      <div id ="module">
        <video id="webcam" ref={videoRef} width="900" height="700" autoPlay muted></video>
        <canvas id="movenet_Canvas" ref={canvasRef} width="900" height="700"></canvas>
      </div>
    </div>
    
  );
};

export default Movenet;
