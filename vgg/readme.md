1. 깊이를 증가하면 정확도가 좋아짐.

2. 3x3 filter를 여러 겹 사용하여 5x5, 7x7 filter를 분해하면 추가적인 비선형성을 부여하고 parameter의 수를 감소시킴.

3. pre-initialization을 이용하면 모델이 빠르게 수렴시킴.

4. data augmentation(resize, crop, flip)을 적용하면 다양한 scale로 feature를 포착할 수 있음.

5. 빠른 학습을 위해 4-GPU data parallerism을 활용.
