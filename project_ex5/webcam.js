let detector;
let detectorConfig;
let poses;
let video;
let skeleton = true;
let model;
let squatCount = 0;
let prevHipY = 0;
let squatState = 'up';




async function init() {
  detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER };
  detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
  edges = {
    '5,7': 'm',
    '7,9': 'm',
    '6,8': 'c',
    '8,10': 'c',
    '5,6': 'y',
    '5,11': 'm',
    '6,12': 'c',
    '11,12': 'y',
    '11,13': 'm',
    '13,15': 'm',
    '12,14': 'c',
    '14,16': 'c'
  };
  await getPoses();
}

async function videoReady() {
  //console.log('video ready');
}

async function setup() {
  speakMessage("로딩중입니다")
  createCanvas(640, 480);
  video = createCapture(VIDEO, videoReady);
  //video.size(960, 720);
  video.hide()

  await init();
}

async function getPoses() {
  poses = await detector.estimatePoses(video.elt);
  setTimeout(getPoses, 0);
  //console.log(poses);
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
    // Display squat count
    text(`Squats: ${squatCount}`, 100, 90);
    evaluateSquatForm();
  } else {
    text('Loading, please wait...', 100, 90);
  }
}


function drawKeypoints() {
  var count = 0;
  if (poses && poses.length > 0) {
    for (let kp of poses[0].keypoints) {
      const { x, y, score } = kp;
      if (score > 0.3) {
        count = count + 1;
        fill(255);
        stroke(0);
        strokeWeight(4);
        circle(x, y, 16);
      }
      if (count == 17) {
        //console.log('Whole body visible!');
      }
      else {
        //console.log('Not fully visible!');
      }
    }
  }
}

// Draws lines between the keypoints
function drawSkeleton() {
  const confidence_threshold = 0.5;
  let allLandmarksVisible = true;

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
        strokeWeight(2);
        stroke('rgb(0, 255, 0)');
        line(x1, y1, x2, y2);
      } else {
        allLandmarksVisible = false;
      }
    }

    if (allLandmarksVisible) {
      const leftHip = poses[0].keypoints[11];
      const leftKnee = poses[0].keypoints[13];
      const leftAnkle = poses[0].keypoints[15];
      const rightHip = poses[0].keypoints[12];
      const rightKnee = poses[0].keypoints[14];
      const rightAnkle = poses[0].keypoints[16];

      const leftKneeAngle = angleBetweenThreePoints(leftHip, leftKnee, leftAnkle);
      const rightKneeAngle = angleBetweenThreePoints(rightHip, rightKnee, rightAnkle);

      if (squatState === "up" && leftKneeAngle < 160 && rightKneeAngle < 160) {
        squatState = "down";
      } else if (squatState === "down" && leftKneeAngle > 170 && rightKneeAngle > 170) {
        squatState = "up";
        squatCount++;
      }
    }
  }
}

function angleBetweenThreePoints(a, b, c) {
  const ab = Math.sqrt(Math.pow(b.x - a.x, 2) + Math.pow(b.y - a.y, 2));
  const bc = Math.sqrt(Math.pow(b.x - c.x, 2) + Math.pow(b.y - c.y, 2));
  const ac = Math.sqrt(Math.pow(c.x - a.x, 2) + Math.pow(c.y - a.y, 2));
  return Math.acos((ab * ab + bc * bc - ac * ac) / (2 * ab * bc)) * (180 / Math.PI);
}

function evaluateSquatForm() {
  // 포즈가 올바르게 감지되었는지 확인
   if (poses[0].keypoints.length == 17) {
    const keypoints = poses[0].keypoints;

    // 필요한 keypoints를 가져옵니다.
    const leftShoulder = keypoints[5];
    const rightShoulder = keypoints[6];
    const leftHip = keypoints[11];
    const rightHip = keypoints[12];
    const leftKnee = keypoints[13];
    const rightKnee = keypoints[14];
    const leftAnkle = keypoints[15];
    const rightAnkle = keypoints[16];

    // 등의 기울기를 계산합니다.
    const shoulderSlope = Math.abs(leftShoulder.y - rightShoulder.y);
    const hipSlope = Math.abs(leftHip.y - rightHip.y);

    // 골반의 기울기가 너무 크지 않은지 확인합니다.
    if (hipSlope > 20) {
      speakMessage("골반이 기울어 졌습니다.");
      return "Keep your hips level.";
    }

    // 어깨가 무릎보다 뒤에 있는지 확인합니다.
    if (leftShoulder.x > leftKnee.x || rightShoulder.x > rightKnee.x) {
      speakMessage("어깨가 너무 나갔습니다.");
      return "Lean forward more.";
    }

    // 어깨와 골반 사이에 있는지 확인하여 등을 곧게 유지합니다.
    if (shoulderSlope > 20) {
      speakMessage("등이 굽었습니다.");
      return "Keep your back straight.";
    }

    // 모든 조건이 충족되면 정상적인 스쿼트로 간주합니다.
    speakMessage("좋은 자세입니다.");
    return "Good form!";
  }
  return "";
}

function speakMessage(message) {
  const msg = new SpeechSynthesisUtterance(message);
  msg.lang = 'ko-KR';
  window.speechSynthesis.speak(msg);
}