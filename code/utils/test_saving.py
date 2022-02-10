import numpy as np
import os
import time
import argparse

from utils import default_paths

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--rndval", type=int,default=0,
                    help="rndval")
   
    args = parser.parse_args()

    def save_all(fn2save):
        
        dict2save = {}
        for nn,arr in zip(names, arrays):
            dict2save.update({nn: arr})
            
        if names2 is not None:
            for nn,arr in zip(names2, arrays2):
                dict2save.update({nn: arr})
        if names3 is not None:
            for nn,arr in zip(names3, arrays3):
                dict2save.update({nn: arr})

        print('dict has %d keys'%len(dict2save.keys()))
        print('\nSaving to %s\n'%fn2save)
        np.save(fn2save, dict2save, allow_pickle=True)
        
        
    if args.rndval==0:
        rndval = int(time.strftime('%M%H%d', time.localtime()))  
    else:
        rndval = args.rndval
    fn2save = os.path.join(default_paths.root, 'imStat', 'big_tst_files', 'test_%d.npy'%rndval)
    

    names = ['a', 'b', 'c','d','e']
    arrays = [np.random.normal(0,1,[100,200,100]) for nn in names]
    names2 = None; names3=None
    
    save_all(fn2save)

    print('loading from %s'%fn2save) 
    out = np.load(fn2save, allow_pickle=True).item()
    names=[]; arrays=[];
    for kk in out.keys():
        names.append(kk)
        arrays.append(out[kk])
        
    names2 = ['f', 'g']
    arrays2 = [np.random.normal(0,1,[100,200,100]) for nn in names2]   
      
    save_all(fn2save)
    
    print('loading from %s'%fn2save) 
    out = np.load(fn2save, allow_pickle=True).item()
    names=[]; arrays=[];
    for kk in list(out.keys())[0:5]:
        names.append(kk)
        arrays.append(out[kk])
    names2=[]; arrays2=[];
    for kk in list(out.keys())[5:7]:
        names2.append(kk)
        arrays2.append(out[kk])
       
    names3 = ['h', 'i']
    arrays3 = [np.random.normal(0,1,[100,200,100]) for nn in names3]   
   
    save_all(fn2save)
