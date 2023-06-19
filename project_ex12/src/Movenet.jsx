import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
import '@tensorflow/tfjs-backend-webgl';
import './style.css';

const Movenet = () => {
  const detectorRef = useRef(null);
  const detectorConfigRef = useRef(null);
  const posesRef = useRef(null);
  const skeleton = true;
  const confidenceThreshold = 0.6;
  let video, ctx, canvas;
  let hurrycheckpoint;
  let hurrycheck;
  let repsPerSet = 10; // 세트 개수
  let squatStarted = false;
  let squatFinished = false;
  let kneeAngleThreshold = 130;
  let orangeHurryAngleThreshold = 30;
  let redHurryAngleThreshold = 40;
  let currentSet = 1;
  let totalSets = 5;

  const [timer, setTimer] = useState(0);
  const timerRef = useRef(null);
  const [isPoseDetected, setIsPoseDetected] = useState(false);
  const [isOrangeDetected, setIsOrangeDetected] = useState(false);
  const [isRedDetected, setIsRedDetected] = useState(false);
  const startTimeRef = useRef(null);
  const [squatCount, setSquatCount] = useState(0);

  const videoWidth = 1152;
  const videoHeight = 864;
  const moduleRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

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

  const hurryCheckEdges = {
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

  const hurryErrorEdges = {
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
      startTimeRef.current = Date.now() - timer * 1000; // Subtract elapsed time from current time
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
    detectorConfigRef.current = {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
      enableSmoothing: true,
      enableSegmentation: false,
    };
    detectorRef.current = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      detectorConfigRef.current
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
    posesRef.current = await detectorRef.current.estimatePoses(video);
    setTimeout(getPoses, 0);

    if (posesRef.current && posesRef.current.length > 0) {
      setIsPoseDetected(true);
      const leftShoulder = posesRef.current[0].keypoints[5];
      const rightShoulder = posesRef.current[0].keypoints[6];
      const leftHip = posesRef.current[0].keypoints[11];
      const rightHip = posesRef.current[0].keypoints[12];
      const midShoulderX = (leftShoulder.x + rightShoulder.x) / 2;
      const midShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
      const midHipX = (leftHip.x + rightHip.x) / 2;
      const midHipY = (leftHip.y + rightHip.y) / 2;
      const middleShoulder = {
        x: midShoulderX,
        y: midShoulderY,
        score: Math.min(leftShoulder.score, rightShoulder.score),
      };
      posesRef.current[0].keypoints[17] = {
        x: midShoulderX,
        y: midShoulderY,
        score: Math.min(leftShoulder.score, rightShoulder.score),
      };
      const midhip = {
        x: midHipX,
        y: midHipY,
        score: Math.min(leftHip.score, rightHip.score),
      };
      posesRef.current[0].keypoints[20] = {
        x: midHipX,
        y: midHipY,
        score: Math.min(leftHip.score, rightHip.score),
      };
      const x1 = middleShoulder.x;
      const y1 = middleShoulder.y;
      const x2 = posesRef.current[0].keypoints[20].x;
      const y2 = posesRef.current[0].keypoints[20].y;
      const x1_3 = (2 * x1 + x2) / 3;
      const y1_3 = (2 * y1 + y2) / 3;
      const x2_3 = (x1 + 2 * x2) / 3;
      const y2_3 = (y1 + 2 * y2) / 3;
      posesRef.current[0].keypoints[18] = {
        x: x1_3,
        y: y1_3,
        score: Math.min(leftShoulder.score, rightShoulder.score),
      };
      posesRef.current[0].keypoints[19] = {
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
    } else {
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

    if (posesRef.current && posesRef.current.length > 0) {
      countSquats();
    } else {
    }
    ctx.restore();
    window.requestAnimationFrame(draw);
  };

  const formatTime = (time) => {
    const hours = Math.floor(time / 3600);
    const minutes = Math.floor((time % 3600) / 60);
    const seconds = Math.floor(time % 60);

    const formattedHours = String(hours).padStart(2, '0');
    const formattedMinutes = String(minutes).padStart(2, '0');
    const formattedSeconds = String(seconds).padStart(2, '0');

    return `${formattedHours}:${formattedMinutes}:${formattedSeconds}`;
  };

  const drawKeypoints = () => {
    let count = 0;
    if (posesRef.current && posesRef.current.length > 0) {
      const canvasWidth = canvas.width;
      const canvasHeight = canvas.height;
      const originalWidth = video.videoWidth;
      const originalHeight = video.videoHeight;
      const widthRatio = canvasWidth / originalWidth;
      const heightRatio = canvasHeight / originalHeight;
      ctx.save();

      ctx.font = '10px Arial';

      for (let kp of posesRef.current[0].keypoints) {
        const { x, y, score } = kp;
        const adjustedX = x * widthRatio;
        const adjustedY = y * heightRatio;
        if (score > confidenceThreshold) {
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
    setIsOrangeDetected(false);
    setIsRedDetected(false);
    if (posesRef.current && posesRef.current.length > 0) {
      const canvasWidth = canvas.width;
      const canvasHeight = canvas.height;
      const originalWidth = video.videoWidth;
      const originalHeight = video.videoHeight;
      const widthRatio = canvasWidth / originalWidth;
      const heightRatio = canvasHeight / originalHeight;
      if (hurrycheck > orangeHurryAngleThreshold && hurrycheck <= redHurryAngleThreshold) {
        setIsOrangeDetected(true);
        setIsRedDetected(false);
        for (const [key, value] of Object.entries(hurryCheckEdges)) {
          const p = key.split(',');
          const p1 = parseInt(p[0]);
          const p2 = parseInt(p[1]);
          const kp1 = posesRef.current[0].keypoints[p1];
          const kp2 = posesRef.current[0].keypoints[p2];
          const x1 = kp1.x * widthRatio;
          const y1 = kp1.y * heightRatio;
          const c1 = kp1.score;
          const x2 = kp2.x * widthRatio;
          const y2 = kp2.y * heightRatio;
          const c2 = kp2.score;
          if (c1 > confidenceThreshold && c2 > confidenceThreshold) {
            ctx.strokeStyle = 'rgb(0, 255, 0)'; // 초록색
            ctx.lineWidth = 6;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
        for (const [key, value] of Object.entries(hurryErrorEdges)) {
          const p = key.split(',');
          const p1 = parseInt(p[0]);
          const p2 = parseInt(p[1]);
          const kp1 = posesRef.current[0].keypoints[p1];
          const kp2 = posesRef.current[0].keypoints[p2];
          const x1 = kp1.x * widthRatio;
          const y1 = kp1.y * heightRatio;
          const c1 = kp1.score;
          const x2 = kp2.x * widthRatio;
          const y2 = kp2.y * heightRatio;
          const c2 = kp2.score;
          if (c1 > confidenceThreshold && c2 > confidenceThreshold) {
            ctx.strokeStyle = 'rgb(255, 165, 0)'; // 주황색
            ctx.lineWidth = 6;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
      } else if (hurrycheck > redHurryAngleThreshold) {
        setIsOrangeDetected(false);
        setIsRedDetected(true);
        for (const [key, value] of Object.entries(hurryCheckEdges)) {
          const p = key.split(',');
          const p1 = parseInt(p[0]);
          const p2 = parseInt(p[1]);
          const kp1 = posesRef.current[0].keypoints[p1];
          const kp2 = posesRef.current[0].keypoints[p2];
          const x1 = kp1.x * widthRatio;
          const y1 = kp1.y * heightRatio;
          const c1 = kp1.score;
          const x2 = kp2.x * widthRatio;
          const y2 = kp2.y * heightRatio;
          const c2 = kp2.score;
          if (c1 > confidenceThreshold && c2 > confidenceThreshold) {
            ctx.strokeStyle = 'rgb(0, 255, 0)'; // 초록색
            ctx.lineWidth = 6;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
        for (const [key, value] of Object.entries(hurryErrorEdges)) {
          const p = key.split(',');
          const p1 = parseInt(p[0]);
          const p2 = parseInt(p[1]);
          const kp1 = posesRef.current[0].keypoints[p1];
          const kp2 = posesRef.current[0].keypoints[p2];
          const x1 = kp1.x * widthRatio;
          const y1 = kp1.y * heightRatio;
          const c1 = kp1.score;
          const x2 = kp2.x * widthRatio;
          const y2 = kp2.y * heightRatio;
          const c2 = kp2.score;
          if (c1 > confidenceThreshold && c2 > confidenceThreshold) {
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
          setIsOrangeDetected(false);
          setIsRedDetected(false);
          const p = key.split(',');
          const p1 = parseInt(p[0]);
          const p2 = parseInt(p[1]);
          const kp1 = posesRef.current[0].keypoints[p1];
          const kp2 = posesRef.current[0].keypoints[p2];
          const x1 = kp1.x * widthRatio;
          const y1 = kp1.y * heightRatio;
          const c1 = kp1.score;
          const x2 = kp2.x * widthRatio;
          const y2 = kp2.y * heightRatio;
          const c2 = kp2.score;
          if (c1 > confidenceThreshold && c2 > confidenceThreshold) {
            ctx.strokeStyle = 'rgb(0, 255, 0)'; // 초록색
            ctx.lineWidth = 6;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
      }
    } else {
      setIsOrangeDetected(false);
      setIsRedDetected(false);
    }
  };

  const countSquats = () => {
    const kneeKeypointsConfident = posesRef.current[0].keypoints
      .slice(10, 16)
      .every((kp) => kp.score > confidenceThreshold);
    if (kneeKeypointsConfident) {
      const leftHip = posesRef.current[0].keypoints[11];
      const rightHip = posesRef.current[0].keypoints[12];
      const leftKnee = posesRef.current[0].keypoints[13];
      const rightKnee = posesRef.current[0].keypoints[14];
      const leftAnkle = posesRef.current[0].keypoints[15];
      const rightAnkle = posesRef.current[0].keypoints[16];
      const leftKneeAngle = calculateAngle(leftHip, leftKnee, leftAnkle);
      const rightKneeAngle = calculateAngle(rightHip, rightKnee, rightAnkle);
      if (!squatStarted && leftKneeAngle <= kneeAngleThreshold && rightKneeAngle <= kneeAngleThreshold) {
        squatStarted = true;
        squatFinished = false;
      }
      if (squatStarted && leftKneeAngle > kneeAngleThreshold && rightKneeAngle > kneeAngleThreshold) {
        setSquatCount((prevCount) => prevCount + 1);
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
      <div id="state" style={{ width: videoWidth, height: videoHeight }}>
        <div id="count">
          <span id="A">개수</span><br />
          <span id="B">{squatCount}</span><br />
          <span id="C">/ {repsPerSet}</span>
        </div>
        <div id="shapeContainer">
          {Array.from({ length: totalSets }, (_, index) => (
            <div
              key={index}
              className={index < currentSet ? 'shape completed' : 'shape'}
            ></div>
          ))}
        </div>
        <div id="timer">
          <span id="D">운동 시간 </span>
          <br />
          <span id="E">{formatTime(timer)}</span>
        </div>
        <div id="checkpose">
          <div id="yes1" style={{ display: isPoseDetected ? 'block' : 'none' }}></div>
          <div id="no1" style={{ display: isPoseDetected ? 'none' : 'block' }}>포즈감지가 불안정합니다.</div>
          <div id="yes2" style={{ display: isOrangeDetected ? 'none' : 'block' }}></div>
          <div id="no2" style={{ display: isOrangeDetected ? 'block' : 'none' }}>허리가 굽혀집니다 주의해주세요</div>
          <div id="yes3" style={{ display: isRedDetected ? 'none' : 'block' }}></div>
          <div id="no3" style={{ display: isRedDetected ? 'block' : 'none' }}>허리가 너무굽혀졌습니다.</div>
        </div>
        <div id="close"><button id="closebtn">종료</button></div>
      </div>
      <div id="module" ref={moduleRef}>
        <video id="webcam" ref={videoRef} width={videoWidth} height={videoHeight} autoPlay muted></video>
        <canvas id="movenet_Canvas" ref={canvasRef} width={videoWidth} height={videoHeight}></canvas>
      </div>
    </div>
  );
};

export default Movenet;