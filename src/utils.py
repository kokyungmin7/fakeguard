"""
얼굴 디텍션 및 Crop 통합 모듈
YOLOv8 모델을 사용한 얼굴 감지 및 이미지/영상 전처리
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

# ============================================================================
# 상수 정의
# ============================================================================

# 모델 관련
DEFAULT_MODEL_NAME = "yolov8n-face"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25

# 정렬 관련
EXPANSION_RATIO = 0.5  # 얼굴 영역 확장 비율 (정렬을 위한 여유 공간)
MIN_KEYPOINTS_FOR_ALIGNMENT = 2  # 정렬에 필요한 최소 keypoint 수

# 얼굴 landmark 인덱스 (YOLO face 모델 기준)
RIGHT_EYE_INDEX = 0
LEFT_EYE_INDEX = 1


# ============================================================================
# 데이터 클래스
# ============================================================================


@dataclass
class FacialAreaRegion:
    """얼굴 영역 정보를 담는 데이터 클래스"""

    x: int  # 좌상단 x 좌표
    y: int  # 좌상단 y 좌표
    w: int  # 너비
    h: int  # 높이
    left_eye: Optional[Tuple[int, int]] = None  # 왼쪽 눈 좌표
    right_eye: Optional[Tuple[int, int]] = None  # 오른쪽 눈 좌표
    confidence: Optional[float] = None  # 감지 신뢰도

    @property
    def area(self) -> int:
        """얼굴 영역 면적"""
        return self.w * self.h

    @property
    def has_landmarks(self) -> bool:
        """landmark 정보 존재 여부"""
        return self.left_eye is not None and self.right_eye is not None


# ============================================================================
# YOLO 얼굴 디텍터
# ============================================================================


class YoloFaceDetector:
    """YOLO 모델을 사용한 얼굴 디텍터

    지원 모델: YOLOv8n-face, YOLOv8m-face, YOLOv12s-face 등

    Note:
        - YOLOv8n-face는 keypoints를 제공하여 정렬 가능
        - YOLOv8m-face, YOLOv12s-face는 keypoints 미제공
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        align: bool = True,
        padding: float = 0.0,
    ):
        """
        Args:
            model_path: 모델 파일 절대 경로 (지정 시 model_name 무시)
            model_name: 모델 파일명 (기본값: "yolov8n-face")
            conf_threshold: 얼굴 감지 최소 신뢰도 (0.0 ~ 1.0)
            align: 얼굴 정렬 사용 여부 (기본값: True)
            padding: 얼굴 영역 주변 padding 비율 (0.0 ~ 1.0, 기본값: 0.0)
        """
        if model_path is None:
            model_path = self._get_model_path(model_name)

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.align = align
        self.padding = padding
        self.model = YOLO(model_path)

    @staticmethod
    def _get_model_path(model_name: str) -> str:
        """모델 이름으로부터 모델 파일 경로 생성 (프로젝트 루트 기준)"""
        # src/utils.py에서 프로젝트 루트의 model 디렉토리로 이동
        current_dir = Path(__file__).parent.parent
        return str(current_dir / "model" / f"{model_name}.pt")

    def detect_faces(self, img: NDArray[Any]) -> List[FacialAreaRegion]:
        """이미지에서 얼굴 감지

        Args:
            img: RGB 이미지 (numpy array, HxWx3)

        Returns:
            감지된 얼굴 영역 리스트

        Raises:
            ValueError: 입력 이미지가 유효하지 않은 경우
        """
        if img is None or img.size == 0:
            raise ValueError("입력 이미지가 유효하지 않습니다")

        results = self.model.predict(img, verbose=False, conf=self.conf_threshold)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return []

        return [self._parse_detection(results, i) for i in range(len(results.boxes))]

    def _parse_detection(self, results: Any, index: int) -> FacialAreaRegion:
        """YOLO 결과에서 얼굴 영역 정보 파싱"""
        box = results.boxes.xywh[index].cpu().numpy()
        confidence = float(results.boxes.conf[index].cpu().numpy())

        # 중심 좌표를 좌상단 좌표로 변환
        x_center, y_center, w, h = box
        x = int(x_center - w / 2)
        y = int(y_center - h / 2)

        # keypoints 추출
        left_eye, right_eye = self._extract_keypoints(results, index)

        return FacialAreaRegion(
            x=x,
            y=y,
            w=int(w),
            h=int(h),
            left_eye=left_eye,
            right_eye=right_eye,
            confidence=confidence,
        )

    @staticmethod
    def _extract_keypoints(
        results: Any, index: int
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """YOLO 결과에서 눈 좌표 추출"""
        if results.keypoints is None or len(results.keypoints.xy) <= index:
            return None, None

        keypoints = results.keypoints.xy[index].cpu().numpy()
        if len(keypoints) < MIN_KEYPOINTS_FOR_ALIGNMENT:
            return None, None

        right_eye = tuple(int(coord) for coord in keypoints[RIGHT_EYE_INDEX])
        left_eye = tuple(int(coord) for coord in keypoints[LEFT_EYE_INDEX])

        return left_eye, right_eye

    def _crop_single_face(
        self,
        image: NDArray[Any],
        face: FacialAreaRegion,
        padding: float,
        align: bool,
        return_angle: bool = False,
    ) -> Union[NDArray[Any], Tuple[NDArray[Any], float]]:
        """단일 얼굴을 crop하는 내부 헬퍼 메서드

        Args:
            image: RGB 이미지
            face: 얼굴 영역 정보
            padding: 패딩 비율
            align: 정렬 여부
            return_angle: 각도 반환 여부

        Returns:
            crop된 이미지 또는 (이미지, 각도) 튜플
        """
        x1, y1, x2, y2 = calculate_face_bounds(face, image.shape[:2], padding)

        if align:
            cropped, angle = process_face_alignment(image, face, x1, y1, x2, y2)
        else:
            cropped = image[y1:y2, x1:x2]
            angle = 0.0

        return (cropped, angle) if return_angle else cropped

    def crop_face(
        self,
        image: NDArray[Any],
        padding: Optional[float] = None,
        align: Optional[bool] = None,
    ) -> List[NDArray[Any]]:
        """이미지에서 모든 얼굴을 crop하여 반환

        Args:
            image: RGB 이미지 (numpy array, HxWx3)
            padding: 얼굴 영역 주변 padding 비율 (None이면 인스턴스 기본값 사용)
            align: 얼굴 정렬 사용 여부 (None이면 인스턴스 기본값 사용)

        Returns:
            crop된 얼굴 이미지 리스트

        Example:
            >>> detector = YoloFaceDetector(align=True, padding=0.1)
            >>> faces = detector.crop_face(img_rgb)
        """
        padding = self.padding if padding is None else padding
        align = self.align if align is None else align

        faces = self.detect_faces(image)
        if not faces:
            return []

        return [
            self._crop_single_face(image, face, padding, align, return_angle=False)
            for face in faces
        ]

    def crop_largest_face(
        self,
        image: NDArray[Any],
        padding: Optional[float] = None,
        align: Optional[bool] = None,
        return_angle: bool = False,
    ) -> Union[Optional[NDArray[Any]], Tuple[Optional[NDArray[Any]], float]]:
        """이미지에서 가장 큰 얼굴만 crop하여 반환

        Args:
            image: RGB 이미지 (numpy array, HxWx3)
            padding: 얼굴 영역 주변 padding 비율 (None이면 인스턴스 기본값 사용)
            align: 얼굴 정렬 사용 여부 (None이면 인스턴스 기본값 사용)
            return_angle: 정렬 각도를 함께 반환할지 여부

        Returns:
            crop된 얼굴 이미지 (얼굴 없으면 None) 또는 (이미지, 각도) 튜플

        Example:
            >>> detector = YoloFaceDetector(align=True, padding=0.1)
            >>> face, angle = detector.crop_largest_face(img_rgb, return_angle=True)
        """
        padding = self.padding if padding is None else padding
        align = self.align if align is None else align

        faces = self.detect_faces(image)
        if not faces:
            return (None, 0.0) if return_angle else None

        largest_face = max(faces, key=lambda f: f.area)
        return self._crop_single_face(image, largest_face, padding, align, return_angle)

    def crop_most_confident_face(
        self,
        image: NDArray[Any],
        padding: Optional[float] = None,
        align: Optional[bool] = None,
        return_angle: bool = False,
    ) -> Union[Optional[NDArray[Any]], Tuple[Optional[NDArray[Any]], float]]:
        """이미지에서 confidence가 가장 높은 얼굴만 crop하여 반환

        Args:
            image: RGB 이미지 (numpy array, HxWx3)
            padding: 얼굴 영역 주변 padding 비율 (None이면 인스턴스 기본값 사용)
            align: 얼굴 정렬 사용 여부 (None이면 인스턴스 기본값 사용)
            return_angle: 정렬 각도를 함께 반환할지 여부

        Returns:
            crop된 얼굴 이미지 (얼굴 없으면 None) 또는 (이미지, 각도) 튜플

        Example:
            >>> detector = YoloFaceDetector(align=True, padding=0.1)
            >>> face, angle = detector.crop_most_confident_face(img_rgb, return_angle=True)
        """
        padding = self.padding if padding is None else padding
        align = self.align if align is None else align

        faces = self.detect_faces(image)
        if not faces:
            return (None, 0.0) if return_angle else None

        most_confident_face = max(
            faces, key=lambda f: f.confidence if f.confidence is not None else 0.0
        )
        return self._crop_single_face(
            image, most_confident_face, padding, align, return_angle
        )


# ============================================================================
# 얼굴 정렬(Alignment) 헬퍼 함수
# ============================================================================


def calculate_face_bounds(
    face: FacialAreaRegion,
    img_shape: Tuple[int, int],
    padding: float = 0.0,
) -> Tuple[int, int, int, int]:
    """얼굴 영역의 경계를 계산 (패딩 포함)

    Args:
        face: 얼굴 영역 정보
        img_shape: 이미지 크기 (height, width)
        padding: 추가할 padding 비율 (0.0 ~ 1.0)

    Returns:
        (x1, y1, x2, y2) - 경계 좌표
    """
    h, w = img_shape
    pad_w = int(face.w * padding)
    pad_h = int(face.h * padding)

    x1 = max(0, face.x - pad_w)
    y1 = max(0, face.y - pad_h)
    x2 = min(w, face.x + face.w + pad_w)
    y2 = min(h, face.y + face.h + pad_h)

    return x1, y1, x2, y2


def calculate_rotation_angle(
    left_eye: Tuple[int, int],
    right_eye: Tuple[int, int],
) -> float:
    """두 눈 좌표로부터 회전 각도 계산

    Args:
        left_eye: 왼쪽 눈 좌표 (x, y)
        right_eye: 오른쪽 눈 좌표 (x, y)

    Returns:
        회전 각도 (도)
    """
    dx = left_eye[0] - right_eye[0]
    dy = left_eye[1] - right_eye[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def align_img_wrt_eyes(
    img: NDArray[Any],
    left_eye: Tuple[int, int],
    right_eye: Tuple[int, int],
) -> Tuple[NDArray[Any], float]:
    """눈 좌표를 기준으로 이미지를 수평 정렬

    Args:
        img: 얼굴 이미지 (numpy array)
        left_eye: 왼쪽 눈 좌표 (사람 기준)
        right_eye: 오른쪽 눈 좌표 (사람 기준)

    Returns:
        (정렬된 이미지, 회전 각도)
    """
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img, 0.0

    angle = calculate_rotation_angle(left_eye, right_eye)

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    aligned_img = cv2.warpAffine(
        img,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return aligned_img, angle


def extract_expanded_region(
    img: NDArray[Any],
    facial_area: Tuple[int, int, int, int],
    expansion_ratio: float = EXPANSION_RATIO,
) -> Tuple[NDArray[Any], int, int]:
    """얼굴 영역을 확장하여 추출 (정렬 시 여유 공간 확보)

    Args:
        img: 원본 이미지
        facial_area: 얼굴 영역 (x, y, w, h)
        expansion_ratio: 확장 비율

    Returns:
        (확장된 이미지, x offset, y offset)
    """
    x, y, w, h = facial_area
    img_h, img_w = img.shape[:2]

    # 확장 오프셋 계산
    offset_x = int(expansion_ratio * w)
    offset_y = int(expansion_ratio * h)

    # 확장된 경계 좌표 계산
    x1, y1 = x - offset_x, y - offset_y
    x2, y2 = x + w + offset_x, y + h + offset_y

    # 이미지 경계 내부인 경우 단순 크롭
    if x1 >= 0 and y1 >= 0 and x2 <= img_w and y2 <= img_h:
        return img[y1:y2, x1:x2], offset_x, offset_y

    # 경계를 벗어나는 경우: 패딩이 있는 이미지 생성
    return _extract_with_padding(img, x, y, w, h, offset_x, offset_y, img_h, img_w)


def project_facial_area_after_rotation(
    facial_area: Tuple[int, int, int, int],
    angle: float,
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """회전 후 얼굴 영역 좌표 재계산

    Args:
        facial_area: 얼굴 영역 (x1, y1, x2, y2)
        angle: 회전 각도 (도)
        image_size: 이미지 크기 (height, width)

    Returns:
        회전 후 얼굴 영역 (x1, y1, x2, y2)
    """
    angle = angle % 360
    if angle == 0:
        return facial_area

    height, width = image_size
    x1, y1, x2, y2 = facial_area
    face_w, face_h = x2 - x1, y2 - y1

    # 얼굴 영역 중심을 이미지 중심 기준으로 변환
    center_x = (x1 + x2) / 2 - width / 2
    center_y = (y1 + y2) / 2 - height / 2

    # 회전 변환 적용
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    direction = 1 if angle >= 0 else -1

    new_center_x = center_x * cos_a + center_y * direction * sin_a
    new_center_y = -center_x * direction * sin_a + center_y * cos_a

    # 이미지 좌표계로 변환 후 경계 내로 클리핑
    new_center_x += width / 2
    new_center_y += height / 2

    new_x1 = max(0, int(new_center_x - face_w / 2))
    new_y1 = max(0, int(new_center_y - face_h / 2))
    new_x2 = min(width, int(new_center_x + face_w / 2))
    new_y2 = min(height, int(new_center_y + face_h / 2))

    return new_x1, new_y1, new_x2, new_y2


def _extract_with_padding(
    img: NDArray[Any],
    x: int,
    y: int,
    w: int,
    h: int,
    offset_x: int,
    offset_y: int,
    img_h: int,
    img_w: int,
) -> Tuple[NDArray[Any], int, int]:
    """경계를 벗어나는 경우 패딩이 있는 확장 이미지 생성

    Args:
        img: 원본 이미지
        x, y, w, h: 얼굴 영역 좌표 및 크기
        offset_x, offset_y: 확장 오프셋
        img_h, img_w: 이미지 크기

    Returns:
        (확장된 이미지, x offset, y offset)
    """
    # 클리핑된 영역 추출
    x1_clipped = max(0, x - offset_x)
    y1_clipped = max(0, y - offset_y)
    x2_clipped = min(img_w, x + w + offset_x)
    y2_clipped = min(img_h, y + h + offset_y)

    cropped_region = img[y1_clipped:y2_clipped, x1_clipped:x2_clipped]

    # 검은 배경 확장 이미지 생성
    expanded_h = h + 2 * offset_y
    expanded_w = w + 2 * offset_x
    expanded_img = np.zeros((expanded_h, expanded_w, img.shape[2]), dtype=img.dtype)

    # 크롭된 영역을 적절한 위치에 배치
    start_x = max(0, offset_x - x)
    start_y = max(0, offset_y - y)
    end_y = start_y + cropped_region.shape[0]
    end_x = start_x + cropped_region.shape[1]

    expanded_img[start_y:end_y, start_x:end_x] = cropped_region

    return expanded_img, offset_x, offset_y


def process_face_alignment(
    img: NDArray[Any],
    face: FacialAreaRegion,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Tuple[NDArray[Any], float]:
    """얼굴 정렬 처리 (공통 로직)

    Args:
        img: 원본 이미지
        face: 얼굴 영역 정보
        x1, y1, x2, y2: 얼굴 경계 좌표

    Returns:
        (정렬된 얼굴 이미지, 회전 각도)
    """
    if not face.has_landmarks:
        return img[y1:y2, x1:x2], 0.0

    # 확장된 영역 추출 및 정렬
    w, h = x2 - x1, y2 - y1
    sub_img, offset_x, offset_y = extract_expanded_region(img, (x1, y1, w, h))

    # 상대 좌표로 변환
    rel_left_eye = (face.left_eye[0] - x1 + offset_x, face.left_eye[1] - y1 + offset_y)
    rel_right_eye = (
        face.right_eye[0] - x1 + offset_x,
        face.right_eye[1] - y1 + offset_y,
    )

    # 이미지 정렬
    aligned_img, angle = align_img_wrt_eyes(sub_img, rel_left_eye, rel_right_eye)

    # 정렬 후 얼굴 영역 좌표 재계산 및 크롭
    rotated_coords = project_facial_area_after_rotation(
        facial_area=(offset_x, offset_y, offset_x + w, offset_y + h),
        angle=angle,
        image_size=(aligned_img.shape[0], aligned_img.shape[1]),
    )

    rx1, ry1, rx2, ry2 = rotated_coords
    return aligned_img[ry1:ry2, rx1:rx2], angle


# ============================================================================
# 유틸리티 함수
# ============================================================================


def save_cropped_image(
    img: NDArray[Any],
    save_path: str,
    convert_to_bgr: bool = True,
) -> None:
    """crop된 이미지를 파일로 저장

    Args:
        img: 저장할 이미지 (RGB 또는 BGR)
        save_path: 저장 경로
        convert_to_bgr: RGB → BGR 변환 여부
    """
    if convert_to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img)
