import warnings
warnings.filterwarnings('ignore') #忽视

import pandas as pd

loans_2020 = pd.read_csv('LoanStats3a.csv', skiprows=1) #第一行是字符串，所以要skiprows=1跳过第一行
half_count = len(loans_2020) / 2 # 4万行除以2 = 19767.5行

loans_2020 = loans_2020.dropna(thresh=half_count, axis=1)#2万行中剔除空白值超过一半的列，thresh：剔除
loans_2020 = loans_2020.drop(['desc', 'url'],axis=1) #按照列中，删除描述和URL链接
loans_2020.to_csv('loans_2020.csv', index=False) #追加到“loans_2007.csv”文件 ， index=False表示不加索引

import pandas as pd

loans_2020 = pd.read_csv("loans_2020.csv")


print("第一行的数据展示 \n",loans_2020.iloc[0])  #第一行的数据

#7.8x10^7,简写为“7.8E+07”的形式 1296599

print("原始列数= ",loans_2020.shape[1]) # shape[1]代表有多少列 ；shape[0]代表有多少行

#id：用户ID
#member_id：会员编号
#funded_amnt：承诺给该贷款的总金额
#funded_amnt_inv：投资者为该贷款承诺的总金额
#grade：贷款等级。贷款利率越高，则等级越高
#sub_grade：贷款子等级
#emp_title：工作名称
#issue_d：贷款月份

loans_2020 = loans_2020.drop(["id", "member_id", "funded_amnt", "funded_amnt_inv", "grade", "sub_grade", "emp_title", "issue_d"], axis=1)

#loan_status：Fully Paid——>全部付讫  Charged Off——>没有按时还款  


loans_2020['loan_status']

#zip_code：常用的邮编
#out_prncp和out_prncp_inv都是一样的：总资金中剩余的未偿还本金
#out_prncp_inv：实际未偿还的本金
#total_rec_prncp：迄今收到的本金



loans_2020 = loans_2020.drop(["zip_code", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp"], axis=1)

#total_rec_int：迄今收到的利息

#recoveries：是否收回本金
#collection_recovery_fee：收集回收费用
#last_pymnt_d：最近一次收到还款的时间
#last_pymnt_amnt：全部的还款的时间


#保留候选特征
loans_2020 = loans_2020.drop(["total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt"], axis=1)
print(loans_2020.iloc[0])#第一行数据


print("现在的列数 = ",loans_2020.shape[1]) #原始是52列，现在是32列的候选特征，还是有点“冗余”，但是还少了一个东西呀，是啥样.....是不是“标签label”啊

print(loans_2020['loan_status'].value_counts())#计算该列特征的属性的个数


#Fully Paid：批准了客户的贷款，后面给他打个“1”
#Charged Off：没有批准了客户的贷款，后面给他打个“0”
#Does not meet the credit policy. Status:Fully Paid：，没有满足要求的有1988个，也不要说清楚不贷款，就不要这个属性了
#后面的属性不确定比较强
#Late (16-30 days)  ：延期了16-30 days
#Late (31-120 days)：延期了31-120 days ， 所以这些都不确定的属性，相当于“取保候审”

loans_2020['loan_status']

#要做一个二分类，用0 1 表示
loans_2020 = loans_2020[(loans_2020['loan_status'] == "Fully Paid") |
                        (loans_2020['loan_status'] == "Charged Off")]
status_replace = {
    #特征当做key，value里还有一个字典
    "loan_status": {
        #第一个键值改为1 ，第二个键值改为0
        "Fully Paid": 1, #完全支付
        "Charged Off": 0,#违约
    }
}
#可以用pandas的DataFrame的格式，做成字典

loans_2020 = loans_2020.replace(status_replace)  #replace：执行的是查找并替换的操作

loans_2020['loan_status']

#在原始数据中的特征值或者属性里都是一样的，对于分类模型的预测是没有用的
#某列特征都是n n n  NaN  n n ,有缺失的，唯一的属性就有2个，用pandas空值给去掉

orig_columns = loans_2020.columns  #展现出所有的列

drop_columns = []  #初始化空值

for col in orig_columns:
    #   dropna()先删除空值，再去重算唯一的属性
    col_series = loans_2020[col].dropna().unique()  #去重唯一的属性
    if len(col_series) == 1:  #如果该特征的属性只有一个属性，就给过滤掉该特征
        drop_columns.append(col)
        
loans_2020 = loans_2020.drop(drop_columns, axis=1)
print(drop_columns)
print("--------------------------------------------")
print(loans_2020.shape)
loans_2020.to_csv('filtered_loans_2020.csv', index=False)

#还剩下24个候选特征

import pandas as pd
loans = pd.read_csv('filtered_loans_2020.csv')
null_counts = loans.isnull().sum()  #用pandas的isnull统计一下每列的缺失值，给累加起来
print(null_counts) 

#对于每列中缺失的情况不是很大，大多数是很好的 ，那就删掉几个列也无可厚非(对于样本大)，或者是只删除缺失值，或者用均值、中位数和众数补充

loans = loans.drop("pub_rec_bankruptcies", axis=1)
loans = loans.dropna(axis=0) 
#用dtypes类型统计有多少个是object、int、float类型的特征
print(loans.dtypes.value_counts())

#Pandas里select_dtypes只选定“ohbject”的类型str，只选定字符型的数据

object_columns_df = loans.select_dtypes(include=["object"])
print(object_columns_df.iloc[0])


#term：分期多少个月啊
#int_rate：利息，10.65%，后面还要把%去掉
#emp_length：10年的映射成10，9年的映射成9
#home_ownership：房屋所有权，是租的、还是自己的、还是低压出去了，那就用0 1 2来代替

#查看指定标签的属性，并记数
#home_ownership：房屋所有权
#verification_status：身份保持证明
#emp_length：客户公司名称
#term：贷款分期的时间
#addr_state：地址邮编


cols = [
    'home_ownership', 'verification_status', 'emp_length', 'term', 'addr_state'
]
for c in cols:
    print(loans[c].value_counts())

#"purpose"和"title"表达的意思相近，且从输出结果可以看出"title"所含的属性较多，可以将其舍弃掉
print(loans["purpose"].value_counts())#purpose：你贷款时的目的是什么，买房还是买车，还是其他消费

print("------------------------------------------------")

print(loans["title"].value_counts())#title：跟purpose一样，贷款的目的，选一个就行了

#labelencoder
# jemp_length做成字典，emp_length当做key ，value里还是字典 ，"10+ years": 10...
# 再在后面调用replace函数，刚才利息这列特征，是不是有%啊，再用astype()处理一下

mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}

# 删除：last_credit_pull_d：LC撤回最近的月份   
#earliest_cr_line：第一次借贷时间
#addr_state：家庭邮编
#title：URL的标题
loans = loans.drop(
    ["last_credit_pull_d", "earliest_cr_line", "addr_state", "title"], axis=1)
#rstrip：删除 string 字符串末尾的指定字符
loans["int_rate"] = loans["int_rate"].str.rstrip("%").astype("float")
#revol_util：透支额度占信用比例
loans["revol_util"] = loans["revol_util"].str.rstrip("%").astype("float")
loans = loans.replace(mapping_dict)

mapping_dict

#查看指定标签的属性，并记数
#home_ownership：房屋所有权
#verification_status：身份保持证明
#emp_length：客户公司名称
#purpose：贷款的意图
#term：贷款分期的时间

cat_columns = ["home_ownership", "verification_status", "emp_length", "purpose", "term"]
dummy_df = pd.get_dummies(loans[cat_columns])
#concat() 方法用于连接两个或多个数组,
loans = pd.concat([loans, dummy_df], axis=1)
loans = loans.drop(cat_columns, axis=1)
#pymnt_plan 指示是否已为贷款实施付款计划 ，里面都为N，删掉这一列
loans = loans.drop("pymnt_plan", axis=1)

loans.to_csv('cleaned_loans_2020.csv', index=False)

import pandas as pd
loans = pd.read_csv("cleaned_loans_2020.csv") # 清洗完的数据拿过来，现在的数据要么是float类型和int类型
print(loans.info())

#到了这步了，数据已经处理好了，开始咱们喜欢玩的机器学习Machine Learning
#LR不是回归而是分类，用它进行训练了
from sklearn.linear_model import LogisticRegression # 分类
lr = LogisticRegression() # 调用逻辑回归的算法包



cols = loans.columns # 4万行 * 24列的样本
train_cols = cols.drop("loan_status") # 删除loan_status这一列作为目标值

features = loans[train_cols] # 23列的特征矩阵
target = loans["loan_status"] # 作为标签矩阵

lr.fit(features, target) #开始训练
predictions = lr.predict(features) # 开始预测

predictions[:10] #0:代表没有偿还  1:代表偿还

lr.predict_proba(features)#lr的概率模型

lr

import pandas as pd
#接下来就是如何算4个指标 fp tp fn tn

# 假正类（False Positive，FP）：将负类预测为正类
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])
print(fp)
print("----------------------------------------")


# 真正类（True Positive，TP）：将正类预测为正类
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])
print(tp)
print("----------------------------------------")


# 假负类（False Negative，FN）：将正类预测为负类
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])
print(fn)
print("----------------------------------------")

# 真负类（True Negative，TN）：将负类预测为负类
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])
print(tn)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

lr = LogisticRegression()
predictions = cross_val_predict(lr, features, target, cv=10) # Kfold = 10(交叉验证)
predictions = pd.Series(predictions)
print(predictions[:1000])

# 假正类（False Positive，FP）：将负类预测为正类
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])


# 真正类（True Positive，TP）：将正类预测为正类
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])



# 假负类（False Negative，FN）：将正类预测为负类
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])
#


# 真负类（True Negative，TN）：将负类预测为负类
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates：就可以用刚才的指标进行衡量了呀
tpr = tp / float((tp + fn))
fpr = fp / float((fp + tn))
"""
tpr:比较高，我们非常喜欢，给他贷款了，而且这些人能还钱了
fpr：比较高，这些人不会还钱，但还是贷给他了吧
为什么这个2个值都那么高呢？把所有人来了，都借给他钱呀，打印出前20行都为1，为什么会出现这种情况？
绝对是前面的数据出现问题了，比如说数据是6：1，绝大多数是1，小部分是0，样本不均衡的情况下，导致分类器错误的认为把所有的样本预测为1，因为负样本少，咱们就“数据增强”，
把负样本1增强到4份儿，是不是可以啊，要么收集数据 ，数据已经定值了，没办法收集，要么是造数据，你知道什么样的人会还钱吗？也不好造吧，怎么解决样本不均衡的问题呢？
接下来要考虑权重的东西了，一部分是6份，另一部分是1份，把6份的权重设置为1，把1份的权重设置为6，设置权重项来进行衡量，把不均衡的样本变得均衡，加了权重项，让正样本对结果的影响小一些，
让负样本对结果的影响大一些，通过加入权重项，模型对结果变得均衡一下，有一个参数很重要
"""
print(tpr)#真正率
print(fpr)#假正率
print(predictions[:20])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
"""
class_weight：可以调整正反样本的权重
balanced:希望正负样本平衡一些的
"""
lr = LogisticRegression(class_weight="balanced")
predictions = cross_val_predict(lr, features, target, cv=10)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / float((tp + fn))
fpr = fp / float((fp + tn))

print(tpr)#真正率
print()
print(fpr)#假正率
print()
print(predictions[:20])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
"""
权重项可以自己定义的
0代表5倍的
1代表10倍的
"""
penalty = {
    0: 5,
    1: 1
}

lr = LogisticRegression(class_weight=penalty)
# kf = KFold(features.shape[0], random_state=1)
kf = 10
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / float((tp + fn))
fpr = fp / float((fp + tn))

print(tpr)
print()
print(fpr)

