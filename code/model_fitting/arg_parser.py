import argparse
import numpy as np

def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--volume_space", type=int, default=1,
                    help="want to do fitting with volume space or surface space data? 1 for volume, 0 for surface.")
    parser.add_argument("--up_to_sess", type=int,default=1,
                    help="analyze sessions 1-#")
    
    
    
    parser.add_argument("--fitting_type", type=str,default='texture_pyramid',
                    help="what kind of fitting are we doing? opts are 'texture_pyramid', 'texture_gabor', 'gabor_solo', 'bdcn', 'texture_pyramid_plus_bdcn'")
                        
    parser.add_argument("--ridge", type=int,default=1,
                    help="want to do ridge regression (lambda>0)? 1 for yes, 0 for no")
    parser.add_argument("--zscore_features", type=int,default=1,
                    help="want to z-score each feature right before fitting encoding model? 1 for yes, 0 for no")
    
   
    
    parser.add_argument("--shuffle_images", type=int,default=0,
                    help="want to shuffle the images randomly (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--random_images", type=int,default=0,
                    help="want to use random gaussian values for images (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--random_voxel_data", type=int,default=0,
                    help="want to use random gaussian values for voxel data (control analysis)? 1 for yes, 0 for no")
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    
    parser.add_argument("--do_fitting", type=int,default=1,
                    help="want to do model training? 1 for yes, 0 for no")
    parser.add_argument("--do_val", type=int,default=1,
                    help="want to do model validation? 1 for yes, 0 for no")
    parser.add_argument("--do_varpart", type=int,default=1,
                    help="want to do variance partition? 1 for yes, 0 for no")
    parser.add_argument("--date_str", type=int,default=0,
                    help="what date was the model fitting done (only if you're starting from validation step.)")
    
     
    parser.add_argument("--sample_batch_size", type=int,default=50,
                    help="number of trials to analyze at once when making features (smaller will help with out-of-memory errors)")
    parser.add_argument("--voxel_batch_size", type=int,default=100,
                    help="number of voxels to analyze at once when fitting weights (smaller will help with out-of-memory errors)")
    
    parser.add_argument("--shuff_rnd_seed", type=int,default=0,
                    help="random seed to use for shuffling, when holding out part of training set for lambda selection.")
    
        
    parser.add_argument("--n_prf_sd_out", type=int, default=2, 
                    help="How many pRF stddevs to use in patches? Default is 2")
    
    # Stuff that is specific to 'gabor' or 'texture' models
    parser.add_argument("--n_ori", type=int,default=36,
                    help="number of orientation channels to use")
    parser.add_argument("--n_sf", type=int,default=12,
                    help="number of spatial frequency channels to use")
    parser.add_argument("--group_all_hl_feats", type=int,default=1,
                    help="want to simplify groups of features in texture model? 1 for yes, 0 for no")
    parser.add_argument("--nonlin_fn", type=int,default=0,
                    help="want to apply a nonlinearity to each feature before fitting encoding model? 1 for yes, 0 for no")
    parser.add_argument("--padding_mode", type=str,default='circular',
                    help="how to pad when doing convolution during gabor feature generation? opts are 'circular','reflect','constant','replicate'; default is circular.")
        
        
    # Stuff that is specific to PCA
    parser.add_argument("--do_pca_bdcn", type=int, default=1,
                    help="want to do PCA on BDCN features before fitting? 1 for yes, 0 for no")
    parser.add_argument("--do_pca_st", type=int, default=1,
                    help="want to do PCA on sketch tokens features before fitting? 1 for yes, 0 for no")
    parser.add_argument("--do_pca_pyr_hl", type=int, default=1,
                    help="want to do PCA on higher level texture features before fitting? 1 for yes, 0 for no")
    parser.add_argument("--min_pct_var", type=int,default=95,
                    help="minimum percent var to use when choosing num pcs to retain, default 95")
    parser.add_argument("--max_pc_to_retain", type=int,default=100,
                    help="maximum number of pcs to retain, default 100")
    
    # Stuff that is specific to BDCN model
    parser.add_argument("--map_ind", type=int, default=-1, 
                    help="which map to use in BDCN model? Default is -1 which gives fused map")    
    parser.add_argument("--do_nms", type=int, default=1,
                    help="In BDCN model, want to apply non-maximal suppression to thin edge maps? 1 for yes, 0 for no")
    parser.add_argument("--downsample_factor", type=np.float32, default=1,
                    help="In BDCN model, downsample edge maps before getting feautures? 1 for yes, 0 for no")

    parser.add_argument("--mult_patch_by_prf", type=int, default=1,
                    help="For BDCN and sketch tokens, want to multiply the feature map patch by pRF gaussian? 1 for yes, 0 for no")
    
    # Specific to sketch tokens
    parser.add_argument("--do_avg_pool", type=int, default=1,
                    help="In sketch tokens model, want to apply avg pooling or max pooling across spatial pRF? 1 for avg, 0 for max")
   
    args = parser.parse_args()
    
    # print values of a few key things to the command line...
    if args.debug==1:
        print('USING DEBUG MODE...')
    if args.ridge==1 and 'pca' not in args.fitting_type:
        print('will perform ridge regression for a range of positive lambdas.')
    else:
        print('will fix ridge parameter at 0.0')    
        
    if args.zscore_features==1:
        print('will perform z-scoring of features')
    else:
        print('skipping z-scoring of features')
    
    if args.shuffle_images==1:
        print('\nWILL RANDOMLY SHUFFLE IMAGES\n')
    if args.random_images==1:
        print('\nWILL USE RANDOM NOISE IMAGES\n')
    if args.random_voxel_data==1:
        print('\nWILL USE RANDOM DATA\n')

    
    
    return args
    