import os
from util import *

def merge_ext(home,home2,exts=['png']):

    if not os.path.exists(home2):
        os.mkdir(home2)

    os.chdir(home2)
    
    for ext in exts:
        if not os.path.exists(ext):
            os.mkdir(ext)
        flist = get_filelist(home,[ext])
        i=0

        for i in range(len(flist)):
            fn = flist[i]
            p = fn.split(os.sep)
            fn_new = os.path.join(ext,p[-2]+p[-1])

            print(fn,fn_new)
            os.rename(fn,fn_new)

# transfer the flt folder into png folder
def change_flt(home, home2):
    filelist = get_filelist(home_dir=home,ext=['flt'])
    if not os.path.exists(home2):
        os.mkdir(home2)

    for fn in filelist:
        fn_write = os.path.join(home2,os.path.basename(fn)[:-3]+'png')
        if not os.path.exists(fn_write):
            im = flt2img(fn)
            cv2.imwrite(fn_write,im)
            print(fn,fn_write)

def build_depth_feature(home,filename,target_size = (64,64)):
    tt_depth = np.array([])
    pics = get_filelist(home,['png'])
    for i in range(len(pics)):
        d = cv2.imread(pics[i],cv2.IMREAD_ANYDEPTH)
        d = cv2.resize(d, target_size)
        d = d.reshape(-1,)
        tt_depth = stack_vector(tt_depth,d)
        print(pics[i],tt_depth.shape)
    np.save(filename,tt_depth)
    print(tt_depth.shape)
    
def merge_matrix(home,folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    flist = get_filelist(home,'npz')
    if len(flist)>0:
        keys = np.load(flist[0]).keys()
        for key in keys:
            print('key',key)
            tt_mat = np.array([])
            for i in range(len(flist)):
                #print(flist[i].keys)
                data = np.load(flist[i])
                mat = data[key]
                tt_mat = stack_vector(tt_mat,mat)
            fn = os.path.join(folder,key)
            print(fn, tt_mat.shape)
            np.save(fn,tt_mat)


if __name__=='__main__':
    home = "/playpen/biao/result/0130/result/"
    home2 = "/playpen/biao/result/0130/merged"
    #merge_ext(home,home2,exts=['flt','npz','png'])
    os.chdir(home2)
    change_flt('flt','depth')
    build_depth_feature('depth','tt_depth')
    merge_matrix(home2, home2)
