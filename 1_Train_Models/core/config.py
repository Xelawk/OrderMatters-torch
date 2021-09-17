import os
import sys
import logging

sys.path.append(os.path.dirname(__file__))

logging.basicConfig(stream = sys.stdout, \
                    # filename='logs/gen_train_dataset.log', \
                    #level=logging.DEBUG)
                    level=logging.INFO)

PROGRAMS = ['openssl']
CVE_PROGRAMS = ['CVE_funcs']    # ignore for now
#CVE_FUNC=['linux-2_6_39_ipip6_rcv','linux-2_6_39_acm_probe','linux-2_6_39_xt_alloc_table_info','linux-2_6_39_xfrm6_tunnel_rcv','linux-2_6_39_wdm_in_callback','linux-2_6_39_usbnet_generic_cdc_bind','linux-2_6_39_tcpmss_mangle_packet','linux-4_1_52_tcpmss_mangle_packet','linux-4_1_52_usbnet_generic_cdc_bind','linux-4_1_52_wdm_in_callback','linux-4_1_52_xfrm6_tunnel_rcv','linux-4_1_52_acm_probe','linux-4_1_52_tcf_act_police_dump','linux-4_1_52_tcf_gact_dump']
#PROGRAMS = ['test']
#PROGRAMS = ['coreutils-8.29', 'findutils-4.6.0','binutils-2.30','openssl','busybox']
#RETRAIN_PROGRAM = ['coreutils']
RETRAIN_PROGRAMS = ['busybox']  # ignore for now
#QUERY_FUNC=['libcrypto_so_1_0_CRYPTO_memcmp']
#QUERY_FUNC=['dwarf_display_debug_ranges']
#QUERY_FUNC=['objdump_load_specific_debug_section']
QUERY_FUNC=['sit_mips32_o2_ipip6_rcv','acm_probe']  # ignore for now
# PROGRAMS = ['vulners']
ARCHS = ['all']
#ARCHS = ['arm']
#ARCHS = ['mips32']# ['aarch64', 'powerpc64'] # ['64']
#ARCHS = ['aarch64', 'mips64','x86','x64','arm32','mips32']# ['64']
#ARCHS = ['x86','mips32','arm32']
OPT_LEVELS = ['all'] # ['o0', 'o1', 'o2', 'o3']
TRAIN_DATASET_NUM = 5000
RETRAIN_DATASET_NUM = 10000  # ignore for now

WORD2VEC_EMBEDDING_SIZE = 50

# TRAIN
NUM_EPOCH = 20

# DATA ROOT
DATA_ROOT_DIR = '/Binary_Similarity/Models/Gemini_Vulseeker_Focus/Gemini_Vulseeker_Focus_Data/Train_Modle'
RETRAIN_DATA_ROOT_DIR = '/home1/mwl/BinBox/data/retrain/'   # retrain不管
# FEATURE
FEA_DIR = os.path.join(DATA_ROOT_DIR, 'features')
CFG_DFG_GEMINIFEA_VULSEEKERFEA = 'cfg_dfg_geminifea_vulseekerfea'
I2VFEA = 'i2v_norm1_' + str(WORD2VEC_EMBEDDING_SIZE)  # i2v_normn_size

FEATURE_GEMINI_DIMENSION = 7
FEATURE_VULSEEKER_DIMENSION = 8
FEATURE_I2V_DIMENSION = WORD2VEC_EMBEDDING_SIZE

# DATASET
DATASET_MIN_BLOCK_NUM = 5   # 最少的基本块数量 default:5
DATASET_MAX_BLOCK_NUM = 100 # 最大的基本块数量 default:30

FILENAME_PREFIX = str(TRAIN_DATASET_NUM) + '_[' + \
                str(DATASET_MIN_BLOCK_NUM) + '_' + \
                str(DATASET_MAX_BLOCK_NUM) + ']_[' + \
                str(I2VFEA) + ']_[' + \
                '_'.join(PROGRAMS) + ']_[' + \
                '_'.join(ARCHS) + ']_[' + \
                '_'.join(OPT_LEVELS) + ']'

RETRAIN_FILENAME_PREFIX = str(RETRAIN_DATASET_NUM) + '_[' + \
                str(DATASET_MIN_BLOCK_NUM) + '_' + \
                str(DATASET_MAX_BLOCK_NUM) + ']_[' + \
                str(I2VFEA) + ']_[' + \
                '_'.join(CVE_PROGRAMS) + ']_[' + \
                '_'.join(ARCHS) + ']_[' + \
                '_'.join(OPT_LEVELS) + ']'


DATASET_DIR = os.path.join(DATA_ROOT_DIR, 'datasets')
DATASET_TRAIN = os.path.join(DATASET_DIR, 'train' + FILENAME_PREFIX + '.csv')
DATASET_VALID = os.path.join(DATASET_DIR, 'valid' + FILENAME_PREFIX + '.csv')
DATASET_TEST = os.path.join(DATASET_DIR, 'test' + FILENAME_PREFIX + '.csv')

RETRAIN_DATASET_DIR = os.path.join(RETRAIN_DATA_ROOT_DIR, 'datasets')
RETRAIN_DATASET_TRAIN = os.path.join(RETRAIN_DATASET_DIR, 'cve_retrain_train' + RETRAIN_FILENAME_PREFIX + '.csv')
RETRAIN_DATASET_VALID = os.path.join(RETRAIN_DATASET_DIR, 'cve_retrain_valid' + RETRAIN_FILENAME_PREFIX + '.csv')
RETRAIN_DATASET_TEST = os.path.join(RETRAIN_DATASET_DIR, 'cve_retrain_test' + RETRAIN_FILENAME_PREFIX + '.csv')
# 

FP_RETRAIN_DATASET_TRAIN = os.path.join(RETRAIN_DATASET_DIR, 'fp_retrain_train' + RETRAIN_FILENAME_PREFIX + '.csv')
FP_RETRAIN_DATASET_VALID = os.path.join(RETRAIN_DATASET_DIR, 'fp_retrain_valid' + RETRAIN_FILENAME_PREFIX + '.csv')
FP_RETRAIN_DATASET_TEST = os.path.join(RETRAIN_DATASET_DIR, 'fp_retrain_test' + RETRAIN_FILENAME_PREFIX + '.csv')
# MATFILE (for binbox and orderMatters)
MATFILE_DIR = os.path.join(DATA_ROOT_DIR, 'matfiles')
RETRAIN_MATFILE_DIR = os.path.join(RETRAIN_DATA_ROOT_DIR, 'matfiles')

GEN_MATFILE_I2V_BINBOX = True
MATFILE_I2V_BINBOX_DIR = os.path.join(MATFILE_DIR, 'i2v_binbox')
MATFILE_I2V_BINBOX_TRAIN = os.path.join(MATFILE_I2V_BINBOX_DIR, \
                                'train' + FILENAME_PREFIX + '.mat')
MATFILE_I2V_BINBOX_VALID = os.path.join(MATFILE_I2V_BINBOX_DIR, \
                                'valid' + FILENAME_PREFIX + '.mat')
MATFILE_I2V_BINBOX_TEST = os.path.join(MATFILE_I2V_BINBOX_DIR, \
                                'test' + FILENAME_PREFIX + '.mat')

MATFILE_I2V_ORDERMATTERS_DIR = os.path.join(MATFILE_DIR, 'orderMatters')
MATFILE_I2V_ORDERMATTERS_TRAIN = os.path.join(MATFILE_I2V_ORDERMATTERS_DIR, \
                                'train' + FILENAME_PREFIX + '.mat')
MATFILE_I2V_ORDERMATTERS_VALID = os.path.join(MATFILE_I2V_ORDERMATTERS_DIR, \
                                'valid' + FILENAME_PREFIX + '.mat')
MATFILE_I2V_ORDERMATTERS_TEST = os.path.join(MATFILE_I2V_ORDERMATTERS_DIR, \
                                'test' + FILENAME_PREFIX + '.mat')

# TFRECORD
TFRECORD_DIR = os.path.join(DATA_ROOT_DIR, 'tfrecords')
RETRAIN_TFRECORD_DIR = os.path.join(RETRAIN_DATA_ROOT_DIR, 'tfrecords')

GEN_TFRECORD_GEMINI = True
TFRECORD_GEMINI_DIR = os.path.join(TFRECORD_DIR, 'gemini')

TFRECORD_GEMINI_TRAIN = os.path.join(TFRECORD_GEMINI_DIR, \
                                'train' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_GEMINI_VALID = os.path.join(TFRECORD_GEMINI_DIR, \
                                'valid' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_GEMINI_TEST = os.path.join(TFRECORD_GEMINI_DIR, \
                                'test' + FILENAME_PREFIX + '.tfrecord')

GEN_TFRECORD_I2V_GEMINI = True
TFRECORD_I2V_GEMINI_DIR = os.path.join(TFRECORD_DIR, 'i2v_gemini')

TFRECORD_I2V_GEMINI_TRAIN = os.path.join(TFRECORD_I2V_GEMINI_DIR, \
                                'train' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_I2V_GEMINI_VALID = os.path.join(TFRECORD_I2V_GEMINI_DIR, \
                                'valid' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_I2V_GEMINI_TEST = os.path.join(TFRECORD_I2V_GEMINI_DIR, \
                                'test' + FILENAME_PREFIX + '.tfrecord')

GEN_TFRECORD_VULSEEKER = True
TFRECORD_VULSEEKER_DIR = os.path.join(TFRECORD_DIR, 'vulseeker')

TFRECORD_VULSEEKER_TRAIN = os.path.join(TFRECORD_VULSEEKER_DIR, \
                                'train' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_VULSEEKER_VALID = os.path.join(TFRECORD_VULSEEKER_DIR, \
                                'valid' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_VULSEEKER_TEST = os.path.join(TFRECORD_VULSEEKER_DIR, \
                                'test' + FILENAME_PREFIX + '.tfrecord')

GEN_TFRECORD_I2V_VULSEEKER = True
TFRECORD_I2V_VULSEEKER_DIR = os.path.join(TFRECORD_DIR, 'i2v_vulseeker')

TFRECORD_I2V_VULSEEKER_TRAIN = os.path.join(TFRECORD_I2V_VULSEEKER_DIR, \
                                'train' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_I2V_VULSEEKER_VALID = os.path.join(TFRECORD_I2V_VULSEEKER_DIR, \
                                'valid' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_I2V_VULSEEKER_TEST = os.path.join(TFRECORD_I2V_VULSEEKER_DIR, \
                                'test' + FILENAME_PREFIX + '.tfrecord')

GEN_TFRECORD_FUNCSIM = True
TFRECORD_FUNCSIM_DIR = os.path.join(TFRECORD_DIR, 'funcsim')

TFRECORD_FUNCSIM_TRAIN = os.path.join(TFRECORD_FUNCSIM_DIR, \
                                'train' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_FUNCSIM_VALID = os.path.join(TFRECORD_FUNCSIM_DIR, \
                                'valid' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_FUNCSIM_TEST = os.path.join(TFRECORD_FUNCSIM_DIR, \
                                'test' + FILENAME_PREFIX + '.tfrecord')

GEN_TFRECORD_I2V_FUNCSIM = True
TFRECORD_I2V_FUNCSIM_DIR = os.path.join(TFRECORD_DIR, 'i2v_funcsim')

TFRECORD_I2V_FUNCSIM_TRAIN = os.path.join(TFRECORD_I2V_FUNCSIM_DIR, \
                                'train' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_I2V_FUNCSIM_VALID = os.path.join(TFRECORD_I2V_FUNCSIM_DIR, \
                                'valid' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_I2V_FUNCSIM_TEST = os.path.join(TFRECORD_I2V_FUNCSIM_DIR, \
                                'test' + FILENAME_PREFIX + '.tfrecord')

GEN_TFRECORD_CFG_I2V_FUNCSIM = True
TFRECORD_CFG_I2V_FUNCSIM_DIR = os.path.join(TFRECORD_DIR, 'cfg_i2v_funcsim')

TFRECORD_CFG_I2V_FUNCSIM_TRAIN = os.path.join(TFRECORD_CFG_I2V_FUNCSIM_DIR, \
                                'train' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_CFG_I2V_FUNCSIM_VALID = os.path.join(TFRECORD_CFG_I2V_FUNCSIM_DIR, \
                                'valid' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_CFG_I2V_FUNCSIM_TEST = os.path.join(TFRECORD_CFG_I2V_FUNCSIM_DIR, \
                                'test' + FILENAME_PREFIX + '.tfrecord')


GEN_RETRAIN_TFRECORD_CFG_I2V_FUNCSIM = True
RETRAIN_TFRECORD_CFG_I2V_FUNCSIM_DIR = os.path.join(RETRAIN_TFRECORD_DIR, 'cfg_i2v_funcsim')

RETRAIN_TFRECORD_CFG_I2V_FUNCSIM_TRAIN = os.path.join(RETRAIN_TFRECORD_CFG_I2V_FUNCSIM_DIR, \
                                'cve_retrain_train' + RETRAIN_FILENAME_PREFIX + '.tfrecord')
RETRAIN_TFRECORD_CFG_I2V_FUNCSIM_VALID = os.path.join(RETRAIN_TFRECORD_CFG_I2V_FUNCSIM_DIR, \
                                'cve_retrain_valid' + RETRAIN_FILENAME_PREFIX + '.tfrecord')
RETRAIN_TFRECORD_CFG_I2V_FUNCSIM_TEST = os.path.join(RETRAIN_TFRECORD_CFG_I2V_FUNCSIM_DIR, \
                                'cve_retrain_test' + RETRAIN_FILENAME_PREFIX + '.tfrecord')


FP_RETRAIN_TFRECORD_CFG_I2V_FUNCSIM_TRAIN = os.path.join(RETRAIN_TFRECORD_CFG_I2V_FUNCSIM_DIR, \
                                'fp_retrain_train' + RETRAIN_FILENAME_PREFIX + '.tfrecord')
FP_RETRAIN_TFRECORD_CFG_I2V_FUNCSIM_VALID = os.path.join(RETRAIN_TFRECORD_CFG_I2V_FUNCSIM_DIR, \
                                'fp_retrain_valid' + RETRAIN_FILENAME_PREFIX + '.tfrecord')
FP_RETRAIN_TFRECORD_CFG_I2V_FUNCSIM_TEST = os.path.join(RETRAIN_TFRECORD_CFG_I2V_FUNCSIM_DIR, \
                                'fp_retrain_test' + RETRAIN_FILENAME_PREFIX + '.tfrecord')
#
#

# MODEL
MODEL_DIR = os.path.join(DATA_ROOT_DIR, 'models')
RETRAIN_MODEL_DIR = os.path.join(RETRAIN_DATA_ROOT_DIR, 'models')

MODEL_GEMINI_DIR = os.path.join(MODEL_DIR, 'gemini')

MODEL_I2V_GEMINI_DIR = os.path.join(MODEL_DIR, 'i2v_gemini')

MODEL_VULSEEKER_DIR = os.path.join(MODEL_DIR, 'vulseeker')

MODEL_I2V_VULSEEKER_DIR = os.path.join(MODEL_DIR, 'i2v_vulseeker')

MODEL_BINBOX_DIR = os.path.join(MODEL_DIR, 'binbox')

MODEL_ORDERMATERS_DIR = os.path.join(MODEL_DIR, 'orderMatters')

MODEL_FUNCSIM_DIR = os.path.join(MODEL_DIR, 'funcsim')

MODEL_I2V_FUNCSIM_DIR = os.path.join(MODEL_DIR, 'i2v_funcsim')

MODEL_CFG_I2V_FUNCSIM_DIR = os.path.join(MODEL_DIR, 'cfg_i2v_funcsim')
RETRAIN_MODEL_CFG_I2V_FUNCSIM_DIR = os.path.join(RETRAIN_MODEL_DIR, 'cfg_i2v_funcsim')

# STATIS
STATIS_DIR = os.path.join(DATA_ROOT_DIR, 'statis')
RETRAIN_STATIS_DIR = os.path.join(RETRAIN_DATA_ROOT_DIR, 'statis')

STATIS_GEMINI_DIR = os.path.join(STATIS_DIR, 'gemini')

STATIS_I2V_GEMINI_DIR = os.path.join(STATIS_DIR, 'i2v_gemini')

STATIS_VULSEEKER_DIR = os.path.join(STATIS_DIR, 'vulseeker')

STATIS_I2V_VULSEEKER_DIR = os.path.join(STATIS_DIR, 'i2v_vulseeker')

STATIS_BINBOX_DIR = os.path.join(STATIS_DIR, 'binbox')

STATIS_ORDERMATTERS_DIR = os.path.join(STATIS_DIR, 'orderMatters')

STATIS_FUNCSIM_DIR = os.path.join(STATIS_DIR, 'funcsim')

STATIS_I2V_FUNCSIM_DIR = os.path.join(STATIS_DIR, 'i2v_funcsim')

STATIS_CFG_I2V_FUNCSIM_DIR = os.path.join(STATIS_DIR, 'cfg_i2v_funcsim')
RETRAIN_STATIS_CFG_I2V_FUNCSIM_DIR = os.path.join(RETRAIN_STATIS_DIR, 'cfg_i2v_funcsim')
def config_test_and_create_dirs(*args):
    for fname in args:
        d = fname
        if os.path.isfile(fname):
            d, f = os.path.split(fname)
        if not os.path.exists(d):
            os.makedirs(d)

config_test_and_create_dirs( \
        DATA_ROOT_DIR, \
        FEA_DIR, \
        DATASET_DIR, \
        MATFILE_DIR, \
        MATFILE_I2V_BINBOX_DIR, \
        MATFILE_I2V_ORDERMATTERS_DIR, \
        TFRECORD_DIR, \
        TFRECORD_GEMINI_DIR, \
        TFRECORD_I2V_GEMINI_DIR, \
        TFRECORD_VULSEEKER_DIR, \
        TFRECORD_I2V_VULSEEKER_DIR, \
        TFRECORD_FUNCSIM_DIR, \
        TFRECORD_I2V_FUNCSIM_DIR, \
        MODEL_DIR, \
        MODEL_GEMINI_DIR, \
        MODEL_I2V_GEMINI_DIR, \
        MODEL_VULSEEKER_DIR, \
        MODEL_I2V_VULSEEKER_DIR, \
        MODEL_FUNCSIM_DIR, \
        MODEL_I2V_FUNCSIM_DIR, \
        MODEL_ORDERMATERS_DIR, \
        STATIS_DIR, \
        STATIS_GEMINI_DIR, \
        STATIS_I2V_GEMINI_DIR, \
        STATIS_VULSEEKER_DIR, \
        STATIS_I2V_VULSEEKER_DIR, \
        STATIS_FUNCSIM_DIR, \
        STATIS_I2V_FUNCSIM_DIR, \
    )
