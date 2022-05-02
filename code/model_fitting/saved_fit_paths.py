import os
from utils import default_paths
 
# hard coded names of previously fit models that need to be loaded for later steps of fitting

alexnet_fit_paths = ['S01/alexnet_all_conv_pca/Apr-01-2022_1317_39/all_fit_params.npy', \
                     'S02/alexnet_all_conv_pca/Apr-02-2022_2104_46/all_fit_params.npy', \
                     'S03/alexnet_all_conv_pca/Apr-04-2022_0349_08/all_fit_params.npy',  \
                     'S04/alexnet_all_conv_pca/Apr-05-2022_1052_06/all_fit_params.npy', \
                     'S05/alexnet_all_conv_pca/Apr-07-2022_1401_20/all_fit_params.npy', \
                     'S06/alexnet_all_conv_pca/Apr-10-2022_1650_18/all_fit_params.npy', \
                     'S07/alexnet_all_conv_pca/Apr-11-2022_2255_10/all_fit_params.npy', \
                     'S08/alexnet_all_conv_pca/Apr-13-2022_0045_36/all_fit_params.npy']
alexnet_fit_paths = [os.path.join(default_paths.save_fits_path, aa) for aa in alexnet_fit_paths]

clip_fit_paths = ['S01/clip_RN50_all_resblocks_pca/Dec-12-2021_1407_50/all_fit_params.npy', \
                 'S02/clip_RN50_all_resblocks_pca/Jan-13-2022_1121_18/all_fit_params.npy', \
                 'S03/clip_RN50_all_resblocks_pca/Jan-18-2022_1156_04/all_fit_params.npy', \
                 'S04/clip_RN50_all_resblocks_pca/Jan-23-2022_2137_23/all_fit_params.npy', \
                 'S05/clip_RN50_all_resblocks_pca/Jan-27-2022_1124_04/all_fit_params.npy', \
                 'S06/clip_RN50_all_resblocks_pca/Jan-31-2022_0220_32/all_fit_params.npy', \
                 'S07/clip_RN50_all_resblocks_pca/Feb-03-2022_2258_44/all_fit_params.npy', \
                 'S08/clip_RN50_all_resblocks_pca/Feb-09-2022_1544_10/all_fit_params.npy']
clip_fit_paths = [os.path.join(default_paths.save_fits_path, aa) for aa in clip_fit_paths]

gabor_fit_paths = ['S01/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-04-2022_1525_10/all_fit_params.npy', \
                 'S02/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-04-2022_1759_56/all_fit_params.npy', \
                 'S03/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-04-2022_2035_29/all_fit_params.npy', \
                 'S04/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-05-2022_0511_36/all_fit_params.npy', \
                 'S05/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-05-2022_0718_44/all_fit_params.npy', \
                 'S06/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-05-2022_0947_32/all_fit_params.npy', \
                 'S07/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-05-2022_1224_52/all_fit_params.npy', \
                 'S08/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-05-2022_1437_12/all_fit_params.npy']
gabor_fit_paths = [os.path.join(default_paths.save_fits_path, aa) for aa in gabor_fit_paths]