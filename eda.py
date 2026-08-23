import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = sns.load_dataset('tips')
plt.rcParams['font.family'] = ['Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

#데이터 확인
# print(df.info())
# print(df.isna().sum())
# print(df.duplicated().sum())
# print(df.describe())
# print(df.head())

#전처리 - 결측 없음/중복없음
'''
total_bill	총 계산 금액
tip	팁 금액
sex	성별
smoker	흡연자 여부
day	요일
time	점심/저녁
size	일행 수
'''
# 전처리
df['sex'] = (df['sex']=='Female').astype(int) # Female =1
df['smoker'] = (df['smoker']=='Yes').astype(int) #smoker =1
df['time'] = (df['time']=='Lunch').astype(int) #Lunch =1
df['day'] = df['day'].map({'Sun':0, 'Mon':1, 'Tue':2, 'Wed':3, 'Thu':4, 'Fri':5, 'Sat':6})


#팁(tip)에 영향을 주는 것
# 1. 총 금액과 팁의 관계 (당연히 관련 있을 것)
fig, ax = plt.subplots(figsize=(8, 6.5))
corr = df.corr(numeric_only=True)[['tip']]
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', vmin=-1, vmax=1) #	각 칸에 숫자 자동 표시 #vmin=-1, vmax=1
plt.title('1. Tip과 각 label별 상관계수')
plt.show()
# 총금액, 인원수가 tip 기준 상관계수가 높음. (계산 금액이 클수록, 인원이 많을수록 tip을 많이 낸다.

# 2. 총 금액 band별 팁 평균 (주문수 알 수없음)
df['total_bill_band'] = pd.cut(
    df['total_bill'],
    bins=[0, 10, 20, 30, 40, 60],
    labels=['~10', '10~20', '20~30', '30~40', '40~']
)

print('2. 계산금액 구간별 팁 평균')
print(df.groupby('total_bill_band')['tip'].mean()) #구간별 평균 팁 금액을 보면 상관 관계 결과 확인 가능

# 3. 구성원별 팁 평균
print('3. 구성원별 팁 평균')
print(df.groupby('size')['tip'].mean()) #구간별 평균 팁 금액을 보면 상관 관계 결과 확인 가능, 단 착시가 있어 점유율 기준 1인당 팁 평균을 보려함.

# 4. 식당 방문객 구성원별 점유율
print('4. 식당 방문객 구성원별 점유율')
print(df['size'].value_counts(normalize=True) * 100)  #2~4인 방문이 전체의 대다수(약 94.7%)를 차지

# 5. 구성원별 1인당 팁 평균
print('5. 구성원별 1인당 팁 평균')
df['tip_per_person'] = df['tip'] / df['size']
print(df.groupby('size')['tip_per_person'].mean())
# "인원이 많을수록 팁을 많이 낸다"는 겉보기 관계는, 사실 "인원이 많으면 총 금액도 커진다"는 간접 효과였음.
# 1인당으로 정규화하면 정반대 패턴 — 혼자이거나 소수 인원일 때 1인당 팁이 오히려 더 후함.
# 표본이 높은 2~4을 볼때 소수인원일수록 팁이 높음. 2,3,4인순으로 구성원이 1인당 팁이 가장 높음.

# 6~7. 총금액대별 1인당 팁 평균 << deep dive
# print('6. 계산금액 구간&구성별 점유율')
# print(df[['total_bill_band','size']].value_counts(normalize=True) * 100) #10~20 / 20~30 2명 (55.8% 점유율)
print('6. 계산금액 구간별 1인당 팁 평균')
print(df.groupby('total_bill_band', observed=True)['tip_per_person'].mean()) # 20~30 구간이 2번째로 높음.
print('7. 계산금액 구간&구성별 1인당 팁 평균')
print(df.groupby(['total_bill_band','size'])['tip_per_person'].mean()) # 20~30 구간이면서 표본이 가장 두꺼운 지점(2~3명)에서 1인당 팁이 특히 높게 나타남
