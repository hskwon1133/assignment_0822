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
# print(train.info())
# print('-'.center(100, '-'))
# print(test.info())
# print('-'.center(100, '-'))
# print(train.isna().sum())
# print('-'.center(100, '-'))
# print(test.isna().sum()) #Name Null
#문자열 수치화
# train['Transported']=train['Transported'].astype(int)
# train['VIP']=train['VIP'].fillna(train['VIP'].mode()[0]).astype(int)
# print(train.corr(numeric_only=True)['VIP'])

# 전처리
# 1. null이 없는 건 passengerid뿐
# 2. train Transported bool. int변경 필요
# 3. 냉동수면 고객은 우주선 내 편의시설 결제 금액 X (편의시설 합산 0 => 냉동수면, 냉동수면 => 편의시설 0으로 null 값 전환 가능)
# 4. Passengerid 앞 4자리와 last Name으로 FamilyBand 확인 가능, name이 null 값인 사람도 last name은 알 수 있음
# 5. last name & Passengerid 앞 4자리가 같은 사람들이 cabin이 같은 확률도 확인 필요
# 6. RoomService와 Spa, VRDeck, FoodCourt 사용 고객들이 차원 전송에 상관관계를 가지는 것으로 의심. 그 외 str 컬럼도 숫자화 해서 관계성을 볼 필요가 있음.
#   6-1. Roomservice 이용자는 Foodcourt 이용 가능성 낮음. Transported 높음 Age
#   6-2. FoodCourt 이용자는 Spa, VPDeck 이용 가능성 높음. Age
#   6-3. ShoppingMall 이용자는 RoomService > Food court > Spa 다른 서비스 이용자들 보다 연관 가능성이 낮음. 해당 컬럼 필요 없음 수 있음.
#   6-4. Spa 이용자는 Foodcourt>VRDeck 이용 가능성 높음. Tranported 높음 Age
#   6-5. VRDeck 이용자는 FoodCourt>Spa 이용 가능성 높음. Tranported 높음  Age
# 7. VIP는 돈을 무조건 씀. 단, 냉동수면 고객 제외, FoodCourt > VRdeck / Age > Spa > Roomservice
# 8. 냉동수면 고객들 대부분 다른 차원 전송. 대피 못함 이슈 의심
# 9. 나이대별

for df in [train, test]:
    #PassengerId, 앞 4개 번호로 group찾기.
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('Group')['PassengerId'].transform('size')
    # #LastName만 구분하여 Familyname으로 알아내기.
    # df['LastName'] = df['Name'].str.split(' ').str[-1]
    # #LastName 결측치 채워넣기
    # df['LastName'] = df.groupby('GroupSize')['LastName'].transform(lambda x: x.ffill().bfill()) #ffill 앞에 값으로 채움, bfill 뒤의 값으로 채움.
    # #VIP결측치 채우고, int변경
    # df['VIP'] = train['VIP'].fillna(df['VIP'].mode()[0]).astype(int)
    #HomePlanet 결측치 채우고 수치화
    df['HomePlanet'] =df.groupby('Group')['HomePlanet'].transform(lambda x: x.ffill().bfill())
    df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
    df['HomePlanet'] = df['HomePlanet'].map({'Earth': 1, 'Europa': 2, 'Mars': 3})
    #Cabin 결측치는 Num빼고, Deck과 Side 따로 구분, 결측치 채우고 각각 수치화
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['Side'] = df['Cabin'].str.split('/').str[-1]
    for col in ['Deck', 'Side'] :
        df[col] = df.groupby(['Group'])[col].transform(lambda x: x.ffill().bfill())
        df[col] = df[col].fillna(df[col].mode()[0])
    df['Deck'] = df['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8})
    df['Side'] = df['Side'].map({'P': 1, 'S': 2})
    #Destination 결측치 group 최빈값으로 결측치 최우고 각각 수치화.
    df['Destination'] =df.groupby(['Group'])['Destination'].transform(lambda x: x.ffill().bfill())
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
    df['Destination'] = df['Destination'].map({'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3})
    #편의시설 결측값 구하기
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in spend_cols:
        df.loc[df['CryoSleep'] == True, col] = 0 #bool조건으로 True값 행만 찾기, col은 열 = spend_cols
        df[col] = df.groupby(['Group', 'VIP'])[col].transform(lambda x: x.fillna(x.median()))
    df['Service_fee'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    #냉동수면 고객 결측치 채우고 int 변경
    df['CryoSleep'] = df['CryoSleep'].fillna(df['Service_fee']==0)
    df['CryoSleep'] = df['CryoSleep'].astype(int)

#Transported int 변경
df['Transported'] = train['Transported'].astype(int)

# print(train.isna().sum())
# print(train.head())
# print(train.corr(numeric_only=True)['Transported'])

# #모델
features = ['CryoSleep', 'RoomService', 'Spa', 'VRDeck', 'Deck', 'Side', 'HomePlanet', 'Destination']
X = train[features]
y = train['Transported']
m = RandomForestClassifier(n_estimators=300, random_state = 42, max_depth=9)
m.fit(X, y)
result = m.predict(test[features])
test['Transported'] = result
importance = pd.Series(m.feature_importances_, index=features)
# print(importance.sort_values(ascending=False))

#저장
test[['PassengerId', 'Transported']].to_csv('data/result.csv', index=False)


