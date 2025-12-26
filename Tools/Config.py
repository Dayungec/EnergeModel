EXPORT_TIFF=False  #是否在模型构建好后输出预测栅格

POSITIVE="positive"
if EXPORT_TIFF :
    POSITIVE="positive_all"

#BASE_DIR="data/aba/"
#BASE_DIR="data/cd/"
BASE_DIR="data/zjk/"
#BASE_DIR="data/fire/"

NEG_FILE = 'neg_1.xlsx'
#NEG_FILE = 'neg_2.xlsx'

SHAP_ANA=0

SAMPLE_TRESH=0.5  #均匀采样统计密度的阈值



