let detector;                               // 포즈 감지 모델 인스턴스
let detectorConfig;                         // 포즈 감지 모델에 대한 구성 개체
let poses;                                  // 키포인트 및 해당 위치
let video;                                  // 비디오 캡처
let skeleton = true;                        // 연결선 여부
let lastSpokenTimestamp = 0;                // 메세지의 시간 기본값
const confidence_threshold = 0.5;           // 추적도

let squatCount = 0;                         // 스쿼트 횟수
let squatStarted = false;                   // 스쿼트 동작 시작 여부
let squatFinished = false;                  // 스쿼트 동작 종료 여부

const kneeAngleThreshold = 145;             // 스쿼트 각도

let currentSet = 1;                         // 현재 세트
const repsPerSet = 10;                      // 1세트당 개수
const totalSets = 3;                        // 총 세트 수
let restStarted = false;                    // 휴식 중 여부

let keypointsData = [];                     // 좌표값을 저장

const keypointNames = [                     // 키포인트 별 이름저장
  'nose_x', 'nose_y',
  'leftEye_x', 'leftEye_y',
  'rightEye_x', 'rightEye_y',
  'leftEar_x', 'leftEar_y',
  'rightEar_x', 'rightEar_y',
  'leftShoulder_x', 'leftShoulder_y',
  'rightShoulder_x', 'rightShoulder_y',
  'leftElbow_x', 'leftElbow_y',
  'rightElbow_x', 'rightElbow_y',
  'leftWrist_x', 'leftWrist_y',
  'rightWrist_x', 'rightWrist_y',
  'leftHip_x', 'leftHip_y',
  'rightHip_x', 'rightHip_y',
  'leftKnee_x', 'leftKnee_y',
  'rightKnee_x', 'rightKnee_y',
  'leftAnkle_x', 'leftAnkle_y',
  'rightAnkle_x', 'rightAnkle_y',
  'neck_x','neck_y',                        // 17 keypoint
  'Back_x','Back_y',                        // 18 keypoint
  'Waist_x','Waist_y',                      // 19 keypoint
  'midhip_x','midhid_y'                     // 20 keypoint
];


async function init() {
  detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER };
  detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
  edges = {                               // keypoint의 선을 잇는다.
    '5,7': 'm',
    '5,17': 'm',
    '6,17': 'm',
    '7,9': 'm',
    '6,8': 'c',
    '8,10': 'c',
    '11,13': 'm',
    '13,15': 'm',
    '12,14': 'c',
    '14,16': 'c',
    '17,18': 'c',
    '18,19': 'c',
    '19,20': 'c',
    '12,20': 'c',
    '11,20': 'c'
  };
  await getPoses();
}

async function videoReady() {
  //console.log('video ready');
}

async function setup() {
  koreanspeakMessage("로딩중입니다")
  createCanvas(640, 480);
  video = createCapture(VIDEO, videoReady);
  //video.size(960, 720);
  video.hide()

  await init();
}

async function getPoses() {
  poses = await detector.estimatePoses(video.elt);
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
    poses[0].keypoints[20] = { x: midHipX, y: midHipY, score: Math.min(leftHip.score, rightHip.score) };
    
    const x1 = middleShoulder.x;
    const y1 = middleShoulder.y;
    const x2 = poses[0].keypoints[20].x;
    const y2 = poses[0].keypoints[20].y;

    const x1_3 = (2 * x1 + x2) / 3;
    const y1_3 = (2 * y1 + y2) / 3;
    const x2_3 = (x1 + 2 * x2) / 3;
    const y2_3 = (y1 + 2 * y2) / 3;

    poses[0].keypoints[17] = { x: midShoulderX, y: midShoulderY - 30, score: Math.min(leftShoulder.score, rightShoulder.score) };
    poses[0].keypoints[18] = { x: x1_3, y: y1_3, score: Math.min(leftShoulder.score, rightShoulder.score) };
    poses[0].keypoints[19] = { x: x2_3, y: y2_3, score: Math.min(leftHip.score, rightHip.score) };
    
   
  }
}

function draw() {
  background(220);
  translate(width, 0);
  scale(-1, 1);
  image(video, 0, 0, video.width, video.height);

  // Draw keypoints and skeleton
  drawKeypoints();
  if (skeleton) {
    drawSkeleton();
  }

  // Write text
  fill(255);
  strokeWeight(2);
  stroke(51);
  translate(width, 0);
  scale(-1, 1);
  textSize(40);

  if (poses && poses.length > 0) {
    countSquats();
    // checkPose()
    text(`스쿼트 개수: ${squatCount}`, 20, 60);
    text(`세트: ${currentSet}/3`, 20, 110);
  } else {
    text('로딩중입니다...', 180, 110);
  }
}


function drawKeypoints() {
  var count = 0;
  if (poses && poses.length > 0) {
    let currentKeypoints = [];
    for (let kp of poses[0].keypoints) {
      const { x, y, score } = kp;
      if (score > confidence_threshold) {
        count = count + 1;
        fill(255);
        stroke(0);
        strokeWeight(4);
        circle(x, y, 16);
        currentKeypoints.push({ x, y });
      }
    }
    keypointsData.push(currentKeypoints);

    const currentTime = new Date().getTime();
    const timeSinceLastMessage = currentTime - lastSpokenTimestamp;

    if (count < 17 && timeSinceLastMessage > 300000) { // 300000ms = 5분
      // Whole body not fully visible, asking the user to move away from the camera
      koreanspeakMessage("카메라로부터 떨어져주세요");
      lastSpokenTimestamp = currentTime;
    }
  }
}

// Draws lines between the keypoints
function drawSkeleton() {
  

  const currentTime = new Date().getTime();
  const timeSinceLastMessage = currentTime - lastSpokenTimestamp;

  if (poses && poses.length > 0) {
    for (const [key, value] of Object.entries(edges)) {
      const p = key.split(",");
      const p1 = parseInt(p[0]);
      const p2 = parseInt(p[1]);

      const y1 = poses[0].keypoints[p1].y;
      const x1 = poses[0].keypoints[p1].x;
      const c1 = poses[0].keypoints[p1].score;
      const y2 = poses[0].keypoints[p2].y;
      const x2 = poses[0].keypoints[p2].x;
      const c2 = poses[0].keypoints[p2].score;

      if (c1 > confidence_threshold && c2 > confidence_threshold) {
        stroke('rgb(0, 255, 0)');
        line(x1, y1, x2, y2);
      }
    }
  }
}

function koreanspeakMessage(message) {
  const msg = new SpeechSynthesisUtterance(message);
  msg.lang = 'ko-KR';
  window.speechSynthesis.speak(msg);
}

function countSquats() {
  if (poses && poses.length > 0) {
    // 휴식 중이면 함수에서 빠져나옴
    if (restStarted) {
      return;
    }

    const allKeypointsConfident = poses[0].keypoints.every(kp => kp.score > confidence_threshold);

    if (allKeypointsConfident) {
      // Save current keypoints' coordinates
      let currentKeypoints = [];
      for (let kp of poses[0].keypoints) {
        currentKeypoints.push(kp.x, kp.y);
      }
      keypointsData.push(currentKeypoints);
      
      const leftHip = poses[0].keypoints[11];
      const rightHip = poses[0].keypoints[12];
      const leftKnee = poses[0].keypoints[13];
      const rightKnee = poses[0].keypoints[14];
      const leftAnkle = poses[0].keypoints[15];
      const rightAnkle = poses[0].keypoints[16];

      const leftKneeAngle = angleBetweenThreePoints(leftHip, leftKnee, leftAnkle);
      const rightKneeAngle = angleBetweenThreePoints(rightHip, rightKnee, rightAnkle);

      console.log(`Left knee angle: ${leftKneeAngle.toFixed(2)} degrees`);
      console.log(`Right knee angle: ${rightKneeAngle.toFixed(2)} degrees`);

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

      if (squatCount === repsPerSet && currentSet < totalSets) {
        restStarted = true;
        console.log(`세트 ${currentSet} 완료! 휴식 시간 시작...`);
        setTimeout(() => {
          if (currentSet < totalSets) {
            currentSet++;
            squatCount = 0;
            restStarted = false;
            console.log(`휴식 시간 종료! 세트 ${currentSet} 시작.`);
          }
        }, 10000); // 10초 휴식
      }

      // 모든 세트 종료 후 종료 메시지 출력
      if (currentSet > totalSets) {
        console.log("모든 세트 완료!");
        restStarted = false;
        exportCSV(keypointNames, keypointsData);
      }
    }
  }
}

function exportCSV(keypointNames, keypointsData) {
  // Get the current date and time
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  const hour = String(now.getHours()).padStart(2, '0');
  const minute = String(now.getMinutes()).padStart(2, '0');
  const second = String(now.getSeconds()).padStart(2, '0');

  // Create the file name with the current date and time
  const fileName = `${year}_${month}_${day}_${hour}_${minute}_${second}.csv`;

  // Create the CSV content
  let csvContent = keypointNames.join(',') + '\n';
  for (const row of keypointsData) {
    csvContent += row.join(',') + '\n';
  }

  // Create a blob with the CSV content and download it
  const csvBlob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const csvURL = URL.createObjectURL(csvBlob);
  const link = document.createElement('a');
  link.href = csvURL;
  link.download = fileName;
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}



function angleBetweenThreePoints(a, b, c) {
  const ab = Math.sqrt(Math.pow(b.x - a.x, 2) + Math.pow(b.y - a.y, 2));
  const bc = Math.sqrt(Math.pow(b.x - c.x, 2) + Math.pow(b.y - c.y, 2));
  const ac = Math.sqrt(Math.pow(c.x - a.x, 2) + Math.pow(c.y - a.y, 2));
  return Math.acos((ab * ab + bc * bc - ac * ac) / (2 * ab * bc)) * (180 / Math.PI);
}

