'''
PassengerId: 승객 고유 ID. gggg_pp 형식 — gggg는 함께 여행하는 그룹 번호, pp는 그룹 내 순번. 같은 그룹은 보통 가족이지만 항상 그런 건 아님
HomePlanet:	승객이 출발한 행성 (보통 그 사람이 원래 살던 행성)
CryoSleep:	항해 기간 동안 냉동수면을 선택했는지 여부. 냉동수면 중인 승객은 선실 밖으로 못 나감
Cabin:	선실 번호. deck/num/side 형식 — side는 P(좌현, Port) 또는 S(우현, Starboard)
Destination: 승객이 내릴 예정인 목적지 행성
Age:	승객의 나이
VIP: 항해 중 VIP 서비스를 결제했는지 여부
RoomService, FoodCourt, ShoppingMall, Spa, VRDeck: 우주선 내 각종 고급 편의시설에서 승객이 결제한 금액
Name: 승객의 이름(성+이름)
Transported:	승객이 다른 차원으로 전송되었는지 여부 — 이게 바로 예측해야 할 정답(target)
'''

#우주선 타이타닉호가 시공간 이상현상(spacetime anomaly)과 충돌하는 동안,
#승객이 다른 차원으로 전송되었는지(transported) 여부를 예측

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#결측치확인
print(train.info())
print('-'.center(100, '-'))
print(test.info())
print('-'.center(100, '-'))
print(train.isna().sum())
print('-'.center(100, '-'))
print(test.isna().sum())
train['Transported']=train['Transported'].astype(int)
print(train.corr(numeric_only=True)['Transported'])

# 1. null이 없는 건 passengerid뿐
# 2. train Transported bool. int변경 필요
# 3. 냉동수면 고객은 우주선 내 편의시설 결제 금액 X (편의시설 합산 0 => 냉동수면, 냉동수면 => 편의시설 0으로 null 값 전환 가능)
# 4. Passengerid 앞 4자리와 last Name으로 FamilyBand 확인 가능, name이 null 값인 사람도 last name은 알 수 있음
# 5. last name & Passengerid 앞 4자리가 같은 사람들이 cabin이 같은 확률도 확인 필요
# 6. RoomService와 Spa, VRDeck, FoodCourt 사용 고객들이 차원 전송에 상관관계를 가지는 것으로 의심. 그 외 str 컬럼도 숫자화 해서 관계성을 볼 필요가 있음.
# 7. VIP는 돈을 무조건 씀. 단, 냉동수면 고객 제외
# 8. 냉동수면 고객들 대부분 다른 차원 전송. 대피 못함 이슈 의심
# 9. 나이대별




