let detector;                               // 포즈 감지 모델 인스턴스
let detectorConfig;                         // 포즈 감지 모델에 대한 구성 개체
let poses;                                  // 키포인트 및 해당 위치                                
let skeleton = true;                        // 연결선 여부
const confidence_threshold = 0.5;           // 추적도
let video,ctx,canvas;                       // 비디오, 캔버스

let hurrycheckpoint;                        // 허리 체크 포인트
let hurrycheck;                             // 허리 체크 각도

let squatCount = 0;                         // 스쿼트 횟수
let squatStarted = false;                   // 스쿼트 동작 시작 여부
let squatFinished = false;                  // 스쿼트 동작 종료 여부

const kneeAngleThreshold = 110;             // 스쿼트 각도 범위
const hurryAngleThreshold = 30;             // 허리 각도 범위

let currentSet = 1;                         // 현재 세트
let repsPerSet = 3;                        // 1세트당 개수
let totalSets = 3;                          // 총 세트 수
let restStarted = false;                    // 휴식 중 여부
let restTime = 10;                          // 시간

window.onload = function() {
  setup();
};

const init = async () => {
  tf.setBackend('webgpu');
  detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER };
  detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
  edges = {                               // keypoint의 선을 잇는다.
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
  
  hurry_check_edges={
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
  
  hurry_error_edges= {
    '17,18': 'm',
    '18,19': 'm',
    '19,20': 'm'
  }

  await getPoses();

}

const setup = async () => {
  canvas = document.getElementById('movenet_Canvas'); 
  ctx = canvas.getContext('2d'); 
  video = document.getElementById('webcam'); 
  // Activate the webcam
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function(stream) {
      video.srcObject = stream;
      video.play();
      video.onloadedmetadata = function(e) {
        videoReady();
      };
    })
    .catch(function(err) {
      console.log("An error occurred: " + err);
    });

  init();
  draw();
}

function videoReady() {
  // Set the dimensions of the canvas to match the video.
  
}

const getPoses = async () => {
  poses = await detector.estimatePoses(video);
  
  setTimeout(getPoses, 0);
  if (poses && poses.length > 0) {
    // Add custom keypoints
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

    hurrycheckpoint = {x: midHipX, y: midShoulderY, score: Math.min(midhip.score, middleShoulder.score) };
    hurrycheck = calculateAngle(middleShoulder, midhip, hurrycheckpoint);
    // console.log(`hurrycheck: ${hurrycheck.toFixed(2)} degrees`)
  }
}

const draw = () => {
  if (!ctx) return; // ctx가 아직 초기화되지 않았다면 그림을 그리지 않음
  // Clear the canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Flip the canvas horizontally, to create a mirror effect
  ctx.save();
  
  // Draw the current video frame onto the canvas
  ctx.drawImage(video, 0, 0, video.width, video.height);

  // Draw keypoints and skeleton
  drawKeypoints();
  if (skeleton) {
    drawSkeleton();
  }

  // Restore the original transformation matrix (removing the flip)
  ctx.restore();

  // Write text
  ctx.fillStyle = 'white';
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 2;
  ctx.font = '40px Arial';
  ctx.save();
  
  // This is where you'd add the code to decide what text to draw.
  // For example:
  if (poses && poses.length > 0) {
    countSquats();
    ctx.fillText('Pose detected', 10, 50);
    ctx.strokeText('Pose detected', 10, 50);
    ctx.fillText(`스쿼트 : ${squatCount}`, 10, 120);
    ctx.strokeText(`스쿼트 : ${squatCount}`, 10, 120);
    ctx.fillText(`세트: ${currentSet} / ${totalSets}`, 10, 190);
    ctx.strokeText(`세트: ${currentSet} / ${totalSets}`, 10, 190);
  } else {
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

    for (let kp of poses[0].keypoints) {
      const { x, y, score } = kp;

      // Adjust keypoints based on canvas size
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

    
    if (hurrycheck > hurryAngleThreshold){
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
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }
    }
    else{
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
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }
    }
  }
}



const countSquats = () =>{
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
      // console.log(`leftKneeAngle: ${leftKneeAngle.toFixed(2)} degrees`);
      // console.log(`rightKneeAngle: ${rightKneeAngle.toFixed(2)} degrees`);

      // 스쿼트 동작 시작
      if (!squatStarted && leftKneeAngle <= kneeAngleThreshold && rightKneeAngle <= kneeAngleThreshold) {
        squatStarted = true;
        squatFinished = false;
      }


      // 스쿼트 동작 종료
      if (squatStarted && leftKneeAngle > kneeAngleThreshold && rightKneeAngle > kneeAngleThreshold) {
        squatCount++;
        squatStarted = false;
        squatFinished = true;
      }

       // 스쿼트 개수 출력
      if (squatCount > 0 && squatFinished) {
        console.log(`스쿼트 ${squatCount}회 완료!`);
        squatFinished = false;
      }

      if (squatCount === repsPerSet && currentSet <= totalSets) {
        restStarted = true;
          if (currentSet < totalSets) {
            console.log(`세트 ${currentSet} 완료! 휴식 시간 시작...`);
            setTimeout(() => {
                currentSet += 1;
                squatCount = 0;
                restStarted = false;
                console.log(`휴식 시간 종료! 세트 ${currentSet} 시작.`);
            }, restTime * 1000); // 10초 휴식
          }
          // 모든 세트 종료 후 종료 메시지 출력
          if (currentSet >= totalSets) {
            console.log("모든 세트 완료!");
            restStarted = false;
            exportCSV(keypointNames, keypointsData);
            const data = {
                'count' : (squatCount+((currentSet-1)*repsPerSet)),
                'score' : score,
                'bad' : bad
            };
          }
      }
  }
}



const calculateAngle = (p1, p2, p3) => {
  let dx1 = p1.x - p2.x;
  let dy1 = p1.y - p2.y;
  let dx2 = p3.x - p2.x;
  let dy2 = p3.y - p2.y;
  
  // compute the dot product
  let dot = dx1 * dx2 + dy1 * dy2;
  
  // compute the magnitudes
  let mag1 = Math.sqrt(dx1 * dx1 + dy1 * dy1);
  let mag2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);
  
  // use the dot product formula to find the angle
  let angle = Math.acos(dot / (mag1 * mag2));
  
  // return the angle in degrees
  return angle * (180.0 / Math.PI);
}
