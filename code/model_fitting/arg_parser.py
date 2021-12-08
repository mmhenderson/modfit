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
    parser.add_argument("--single_sess", type=int,default=0,
                    help="analyze just this one session (enter integer)")
    
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which grid of candidate prfs?")
    
    
    parser.add_argument("--fitting_type", type=str,default='texture_pyramid',
                    help="what kind of fitting are we doing? opts are 'texture_pyramid', 'texture_gabor', 'gabor_solo'")
    parser.add_argument("--fitting_type2", type=str,default='',
                    help="additional fitting type, for variance partition?")
    parser.add_argument("--fitting_type3", type=str,default='',
                    help="additional fitting type, for variance partition?")
    
    parser.add_argument("--semantic_discrim_type", type=str,default='',
                    help="if semantic model, what dimension?")
    
    
    parser.add_argument("--ridge", type=int,default=1,
                    help="want to do ridge regression (lambda>0)? 1 for yes, 0 for no")
    parser.add_argument("--zscore_features", type=int,default=1,
                    help="want to z-score each feature right before fitting encoding model? 1 for yes, 0 for no")
    parser.add_argument("--zscore_in_groups", type=int,default=0,
                    help="want to z-score groups of columns together, rather than each individually?")
   
    
    parser.add_argument("--shuffle_images", type=int,default=0,
                    help="want to shuffle the images randomly (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--random_images", type=int,default=0,
                    help="want to use random gaussian values for images (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--random_voxel_data", type=int,default=0,
                    help="want to use random gaussian values for voxel data (control analysis)? 1 for yes, 0 for no")
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--save_pred_data", type=int,default=0,
                    help="want to save trialwise predictions for validation set? 1 for yes, 0 for no")
    
    
    parser.add_argument("--do_fitting", type=int,default=1,
                    help="want to do model training? 1 for yes, 0 for no")
    parser.add_argument("--use_precomputed_prfs", type=int, default=0, 
                    help="want to use prf estimates that were already computed? 1 for yes, 0 for no")
    parser.add_argument("--do_val", type=int,default=1,
                    help="want to do model validation? 1 for yes, 0 for no")
    parser.add_argument("--do_stack", type=int,default=0,
                    help="want to do stacking analysis? 1 for yes, 0 for no")
    parser.add_argument("--do_varpart", type=int,default=1,
                    help="want to do variance partition? 1 for yes, 0 for no")
    parser.add_argument("--do_tuning", type=int,default=1,
                    help="want to estimate tuning based on correlations? 1 for yes, 0 for no")
    parser.add_argument("--do_sem_disc", type=int,default=1,
                    help="want to estimate semantic discriminability? 1 for yes, 0 for no")
    parser.add_argument("--do_roi_recons", type=int,default=0,
                    help="want to do population level recons? 1 for yes, 0 for no")
    parser.add_argument("--do_voxel_recons", type=int,default=0,
                    help="want to do single voxel recons? 1 for yes, 0 for no")
    parser.add_argument("--date_str", type=str,default='',
                    help="what date was the model fitting done (only if you're starting from validation step.)")
    
     
    parser.add_argument("--sample_batch_size", type=int,default=50,
                    help="number of trials to analyze at once when making features (smaller will help with out-of-memory errors)")
    parser.add_argument("--voxel_batch_size", type=int,default=100,
                    help="number of voxels to analyze at once when fitting weights (smaller will help with out-of-memory errors)")
    
    parser.add_argument("--shuff_rnd_seed", type=int,default=0,
                    help="random seed to use for shuffling, when holding out part of training set for lambda selection.")
   
    # general PCA arguments
    parser.add_argument("--min_pct_var", type=int,default=95,
                    help="minimum percent var to use when choosing num pcs to retain, default 95")
    parser.add_argument("--max_pc_to_retain", type=int,default=100,
                    help="maximum number of pcs to retain, default 100")
    

    # Stuff that is specific to 'gabor' or 'texture' models
    parser.add_argument("--n_ori_gabor", type=int,default=4,
                    help="number of orientation channels to use")
    parser.add_argument("--n_sf_gabor", type=int,default=4,
                    help="number of spatial frequency channels to use")
    parser.add_argument("--gabor_nonlin_fn", type=int,default=0,
                    help="want to add nonlinearity to gabor features? 1 for yes, 0 for no")
    
    
    # Stuff that is specific to pyramid model
    parser.add_argument("--n_ori_pyr", type=int,default=4,
                    help="number of orientation channels to use")
    parser.add_argument("--n_sf_pyr", type=int,default=4,
                    help="number of spatial frequency channels to use")
    parser.add_argument("--use_pca_pyr_feats_ll", type=int, default=1,
                    help="want to do PCA on lower level texture features before fitting? 1 for yes, 0 for no")    
    parser.add_argument("--use_pca_pyr_feats_hl", type=int, default=1,
                    help="want to do PCA on higher level texture features before fitting? 1 for yes, 0 for no")
    parser.add_argument("--max_pc_to_retain_pyr_ll", type=int,default=100,
                    help="maximum number of pcs to retain, default 100")
    parser.add_argument("--max_pc_to_retain_pyr_hl", type=int,default=100,
                    help="maximum number of pcs to retain, default 100")
    parser.add_argument("--group_all_hl_feats", type=int,default=0,
                    help="want to simplify groups of features in texture model? 1 for yes, 0 for no")
    
   
    # Specific to sketch tokens
    parser.add_argument("--use_pca_st_feats", type=int, default=0,
                    help="Want to use reduced dim (PCA) version of sketch tokens features?")
    parser.add_argument("--use_lda_st_feats", type=int, default=0,
                    help="Want to use reduced dim (LDA) version of sketch tokens features?")
    parser.add_argument("--lda_discrim_type", type=str, default='animacy',
                    help="What labels are used to compute LDA features?")
                             
    # specific to alexnet
    parser.add_argument("--alexnet_layer_name", type=str, default='', 
                       help="What layer of alexnet to use?")
    parser.add_argument("--alexnet_padding_mode", type=str, default='', 
                       help="What padding mode for alexnet conv layers? default zeros.")
    parser.add_argument("--use_pca_alexnet_feats", type=int, default=0, 
                       help="use reduced-dim version of alexnet features?")

    # specific to CLIP
    parser.add_argument("--clip_layer_name", type=str, default='', 
                       help="What layer of clip to use?")
    parser.add_argument("--clip_model_architecture", type=str, default='RN50', 
                       help="What model architecture used for this version of clip?")
    parser.add_argument("--use_pca_clip_feats", type=int, default=1, 
                       help="use reduced-dim version of clip features?")

    
    args = parser.parse_args()
    
    # print values of a few key things to the command line...
    if args.debug==1:
        print('USING DEBUG MODE...')
    if args.ridge==1:
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
    