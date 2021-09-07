import torch
import numpy as np

"""
Module class that can merge other smaller modules (useful for variance partitioning)
"""

class combined_feature_extractor(torch.nn.Module):
    
    def __init__(self, modules, module_names, do_varpart=False):
        
        super(combined_feature_extractor, self).__init__()
        self.modules = modules
        self.module_names = module_names
        self.do_varpart = do_varpart
        
    def init_for_fitting(self, image_size, models, dtype):
        
        max_features = 0
        for module in self.modules:            
            module.init_for_fitting(image_size, models, dtype)
            max_features += module.max_features
            
        self.max_features = max_features
        
    def clear_maps(self):
        
        for module in self.modules:
            module.clear_maps()
            
    def get_partial_versions(self):
        
        if not hasattr(self, 'max_features'):
            raise RuntimeError('need to run init_for_fitting first')

        n_total_feat = _feature_extractor.max_features
        masks = np.ones((1,n_total_feat), dtype=int)
        names = ['full_combined_model']
            
        if self.do_varpart:

            # going to define "masks" that combine certain sub-sets of models features at a time
            # to be uses for variance partitioning
            feature_start_ind = 0

            for mi, module in enumerate(_feature_extractor.modules):

                # first a version that only includes the features in current module
                new_mask = np.zeros((1, n_total_feat), dtype=int)
                new_mask[0,feature_start_ind:feature_start_ind+module.max_features] = 1
                masks = np.concatenate((masks, new_mask), axis=0)
                names += ['just_' + _feature_extractor.module_names[mi]]

                if len(_feature_extractor.modules)>2:        
                    # next a version that only everything but the features in current module
                    # (note if there are just 2 modules, this would be redundant)
                    new_mask = np.ones((1, n_total_feat), dtype=int)
                    new_mask[0,feature_start_ind:feature_start_ind+module.max_features] = 0
                    masks = np.concatenate((masks, new_mask), axis=0)
                    names += ['leave_out_' + _feature_extractor.module_names[mi]]

                # if the module has any subsets of features defined, will also do partial versions with those subsets only
                module_partial_masks, module_partial_names = module.get_partial_versions()
                if len(module_partial_names)>1:        
                    new_masks = np.zeros((len(module_partial_names)-1, n_total_feat), dtype=int)
                    new_masks[:,feature_start_ind:feature_start_ind+module.max_features] = module_partial_masks[1:]
                    masks = np.concatenate((masks, new_masks), axis=0)
                    names += [_feature_extractor.module_names[mi] + '_' + name + '_no_other_modules'for name in module_partial_names[1:]]

                    if len(module_partial_names)==3:        
                        # for this special case, also adding in some other combinations  
                        # if more than two subsets then this will get too complicated...
                        new_masks = np.ones((len(module_partial_names)-1, n_total_feat), dtype=int)
                        new_masks[:,feature_start_ind:feature_start_ind+module.max_features] = module_partial_masks[1:]
                        masks = np.concatenate((masks, new_masks), axis=0)
                        names += [_feature_extractor.module_names[mi] + '_' + name + '_plus_other_modules' for name in module_partial_names[1:]]

                feature_start_ind += module.max_features

        return masks, names
        
    def forward(self, images, prf_params, prf_model_ind, fitting_mode = True):

        for mi, module in enumerate(self.modules):
            
            features, inds = module(images, prf_params, prf_model_ind, fitting_mode)
            
            if mi==0:
                all_features_concat = features
                feature_inds_defined = inds
            else:
                all_features_concat = torch.cat((all_features_concat, features), axis=1)
                feature_inds_defined = np.concatenate((feature_inds_defined, inds), axis=0)
  
        return all_features_concat, feature_inds_defined