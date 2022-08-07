# MoodangE_ver1
## 참고 git repository
   1. https://github.com/tensorturtle/classy-sort-yolov5

   > https://github.com/ultralytics/yolov5 와 https://github.com/abewley/sort 가 합쳐져 있는 형태
### 오류 & 해결책

1. PyCharm으로 구동해볼 때 classy-sort-yolov5의 ****classy_track****의 58번째 줄의
   **from sort import ***에서 오류가 발생

    **해결책 >** PyCharm 프로젝트 안에서 두개의 프로젝트를 **일반 폴더**가 아닌 **소스** 형태로 바꾸면 정상으로 import 됨

    ![Untitled](interface/error1_1.png)

    **↓** 정상 import 된 화면

    ![Untitled](interface/error1_2.png)
    ####
2. classy-sort-yolov5/yolov5/weights/download_weights.sh의 파일이 정상 작동하지 않아 yolov5s.pt 파일을 정상적으로 다운로드 하지 못함.

    **해결책 >** [https://github.com/ultralytics/yolov5/releases/download/v3.1/yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v3.1/yolov5s.pt)
       를 통해 yolov5s.pt 파일을 다운로드 한뒤에 classy-sort-yolov5/yolov5/weights에 붙여넣음
    ####

3. (python classy_track.py --source 동영상 경로 --view-img) or  (python classy_track.py —source 0)를 실행시 발생하는 오류 (
   —view—img: 영상, 0: 웹캠)

   ![Untitled](interface/error3_1.png)

   **AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor’라는 오류가 발생함.**

   **해결책 >** 출처 : https://github.com/ultralytics/yolov5/issues/6948
   ~\site-packages\torch\nn\modules\upsampling.py에서 154줄의 `recompute_scale_factor=self.recompute_scale_factor` 를
   주석처리하면 정상 작동

   ![Untitled](interface/error3_2.png)

   **↓** 정상 작동하며 Multiple Object Tracking (MOT)가 되는것을 볼 수 있음

   ![Untitled](interface/error3_3.png)
            
       