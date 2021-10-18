"""
Created on Mon July 29 11:43:54 2019

@author: Aaron Earle-Richardson
"""
import argparse
import os
import re
import nipype
import json
from shutil import copy, SameFileError
from multiprocessing import cpu_count
from bids import BIDSLayout, BIDSValidator
from bids.layout import models
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as freesurfer
from nipype.interfaces import afni as afni
#from BIDS_converter.data2bids import tree
from tedana.workflows import t2smap_workflow

os.environ['OMP_NUM_THREADS'] = str(cpu_count()) #tedana resets thread count on import. See https://github.com/ME-ICA/tedana/issues/473 for details

def get_parser(): #parses flags at onset of command
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
        , description=""" 
        preprocess.py is a preprocessing script that allows for the user to substitute in any software packages they want for any process. It takes 
        full advantage of BIDS formatting for intuitive dataset querying via the pybids module. It is recommended that users take a quick look at 
        their tutorial (https://github.com/bids-standard/pybids/blob/master/examples/pybids_tutorial.ipynb). One can also use their 
        documentation (https://bids-standard.github.io/pybids/) for reference.
        ex: python3 preprocess.py -i /media/sf_Ubuntu_files/BIDS -ex s12r17 -verb 
        """
        , epilog="""
            Made by Aaron Earle-Richardson (ae166@duke.edu)
            """)

    parser.add_argument(
        "-i"
        , "--input_dir"
        , required=False
        , default=None
        , help="Input data directory(ies), Default: current directory"
        )
    
    parser.add_argument(
        "-verb"
        , "--verbose"
        , required=False
        , action='store_true' 
        , help="Verbosity",
        )
    
    parser.add_argument(
        "-o"
        , "--output_dir"
        , required=False
        , default=None
        , help="Output BIDS directory, Default: Inside current directory "
        )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-ex',
        '--exclude',
        nargs='*',
        required=False,
        default=None,
        help="""
        Option to exclude scans based on either subject number, scan run number, or echo number. This option will import all files in the BIDS folder
        into the BIDSLayout object except those listed here. For example, if I wanted to import all files except subject 12, run 3, echo 2, I would
        add to the command \"-ex s12r3e2\". If subject 12 has only 3 runs, writing \"-ex s12r1 s12r2 s12r3\" is the same as writing \"-ex s12\".
        Mutually exclusive with the -in option.
        """)
    group.add_argument(
        '-in',
        '--include',
        nargs='*',
        required=False,
        default=None,
        help="""
        Option to include scans based on either subject number, scan run number, or echo number. This option will only import files from the BIDS folder
        to the BIDSLayout object if they are listed here. For example, if I wanted to import only subject 12, run 3, echo 2, I would
        add to the command \"-in s12r3e2\". If subject 12 has only 3 runs, writing \"-in s12r1 s12r2 s12r3\" is the same as writing \"-in s12\".
        Mutually eclusive with the -ex option.
        """)

    return parser

class Preprocessing:
    def __init__(self, input_dir=None, output_dir=None, include=None,
                 exclude=None, verbose=False):
        #sets the .self globalization for self variables
        self.is_verbose = False
        self._input_dir = None
        self._output_dir = None

        self.set_data_dir(input_dir)
        self.set_out_dir(output_dir)
        self.set_verbosity(verbose)
        if self._data_dir is not None:
            self.set_bids(include,exclude)

    def set_verbosity(self,verbosity):
        if verbosity:
            self.is_verbose = True

    def set_bids(self,include,exclude):
        """

        This function sets up the BIDSLayout object by including/excluding
         scans by subject, run, or echo

        :param include:
        :type include:
        :param exclude:
        :type exclude:
        :return:
        :rtype:
        """

        if exclude is not None:
            parsestr = "|".join(exclude)
        elif include is not None:
            parsestr = "|".join(include)
        else:
            parsestr = None

        if parsestr is not None:
            patterns = ""
            for i in range(len(parsestr)):
                if parsestr[i] in "sS":
                    patterns += ".*sub-"
                elif parsestr[i] in "rR":
                    patterns += ".*run-"
                elif parsestr[i] in "eE":
                    patterns += ".*echo-"
                elif parsestr[i] in "0123456789":
                    if i < len(parsestr)-1:
                        if parsestr[i+1] not in "0123456789":
                            if parsestr[i-1] in "0123456789":
                                patterns += parsestr[i-1:i+1]
                            elif parsestr[i] == "0":
                                    ".*".join(patterns.split(".*")[:-1]) 
                            else:
                                patterns += parsestr[i].zfill(2)
                    elif parsestr[i-1] in "0123456789":
                        patterns += parsestr[i-1:i+1] + ".*"
                    elif parsestr[i] not in "123456789":
                        ".*".join(patterns.split(".*")[:-1]) +".*"
                    else:
                        patterns += parsestr[i].zfill(2) + ".*"
                   
        #actually including or excluding files
        ignore = []
        for root, _, files in os.walk(self._data_dir):
            for file in files:
                if exclude is not None and re.match(patterns,file):
                    ignore.append(os.path.join(root,file))
                elif include is not None and not re.match(patterns,file):
                    ignore.append(os.path.join(root,file))
        self.BIDS_layout = BIDSLayout(self._data_dir,ignore=ignore)

    def get_data_dir(self):
        return self._data_dir

    def set_data_dir(self, data_dir): #check if input dir is listed
        if data_dir is None:
            self._data_dir = os.getcwd()
        else:
            self._data_dir = data_dir

    def set_out_dir(self, output_dir): #performs the necessary job of creating a dataset_description.json
        if output_dir is None:
            self._output_dir = os.path.join(self._data_dir,"derivatives/preprocessing")
            if not os.path.isdir(self._output_dir):
                os.makedirs(self._output_dir) 
        else:
            self._output_dir = output_dir

        if not os.path.isfile(os.path.join(self._output_dir,"dataset_description.json")):
            copy(os.path.join(self._data_dir,"dataset_description.json"),os.path.join(self._output_dir,"dataset_description.json")) 
            with open(os.path.join(self._output_dir,"dataset_description.json"),'r') as fst:
                data = json.load(fst)
            with open(os.path.join(self._output_dir,"dataset_description.json"),'w') as fst:
                entry = {}
                entry["PipelineDescription"] = {'Name': 'preprocessing'}
                data.update(entry)
                json.dump(data, fst)

    def FuncHandler(self,fileobj,output,suffix):
        """Allows files to overwrite their inputs as outputs without causing errors


        it also checks root, working directory, and origional file address. It also allows for the user
         to add on a suffix instead of typing a whole path for a new output. Finally, it integrates a sub-brick
         specifier  so that specific subsections of an image may be selected instead of the whole thing.

        :param fileobj:
        :type fileobj:
        :param output:
        :type output:
        :param suffix:
        :type suffix:
        :return:
        :rtype:
        """
        if type(fileobj) == models.BIDSImageFile: #setting file to temp file before performing operation
            fileobj = fileobj.path
        elif type(fileobj) is not str:
            raise TypeError('file inputs must be either a BIDSImageFile, pathlike string')

        smatch = re.match(r"(.*\.nii(?:\.gz))(((?:\[|\{)\d+(?:\.\.\d+|)(?:\]|\})){1,2})",fileobj) #sub-brick parser similar to afni's
        dmatch = re.match(r"(.*\.1D)(((?:\[|\{)\d+(?:\.\.\d+|)(?:\]|\})){1,2})",fileobj) #if you want to understand this mess, go here:
                                                                                         #https://docs.python.org/3/library/re.html
        if smatch:
            fileobj = smatch.group(1).split(".nii.gz")[0]+"_desc-temp.nii.gz"
            afni.TCatSubBrick(in_files=[(smatch.group(1),"'{index}'".format(index=smatch.group(2)))],out_file=fileobj).run()
        elif dmatch:
            fileobj = dmatch.group(1).split(".nii.gz")[0]+"_desc-temp.nii.gz"
            afni.Cat(in_files=[dmatch.group(1)],sel="'{index}'".format(index=dmatch.group(2)),out_file=fileobj).run()

        
        if not os.path.isabs(fileobj) or not os.path.isfile(fileobj): #checking if file exists and address is absolute
            tempfileobj = os.path.abspath(fileobj)
            if not os.path.isfile(tempfileobj) and self._data_dir is not None: #checking working directory for file
                tempfileobj = os.path.join(self.BIDS_layout.root,fileobj)
                if not os.path.isfile(tempfileobj) and self._output_dir is not None: #checiking BIDS root directory for file
                    tempfileobj = os.path.join(self._output_dir, fileobj)
                    if not os.path.isfile(tempfileobj):         #checking BIDS derivatives derectory for file
                        raise FileNotFoundError("could not find {filename} in derivatives, working, or root directory, check naming and try again".format(filename=fileobj))
                    else:
                        fileobj = tempfileobj
                else:
                    fileobj = tempfileobj
            else:
                fileobj = tempfileobj

        if output == None and suffix == None:
            """
            Renames input file to add a "desc-temp" suffix if the user has not given an output file name. 
            This marks the file for deletion but only after the output file (with the same name) is generated,
            effectively "overwrighting" the file. This will not happen if an output is named
            """

            if "_desc-temp" not in fileobj:
                output = fileobj
                fileobj = output.split('.nii.gz')[0] + "_desc-temp.nii.gz"
                os.replace(output,fileobj)
            else:
                output = fileobj.replace("_desc-temp","")
        elif output == None:
            output = fileobj.split('.nii.gz')[0] + "_desc-" + suffix + '.nii.gz'
        elif suffix == None:
            pass
        else:
            print("both suffix and output filename detected as input, using filename given")
        return(fileobj,output)
            
    """These are the standalone tools, useable on their own and customiseable for alternate preprocessing algorithms.
     it is recommended that you not edit anything above this line (excluding package imports and argparser)
     without a serious knowledge of python and this script 
    """

    #cortical reconstruction
    def cortical_recon(self,filepath=None):
        """
        https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.freesurfer/preprocess.html#reconall

        :param filepath:
        :type filepath:
        :return:
        :rtype:
        """
        if filepath == None:
            filepath = self._data_dir
        freesurfer.ReconAll(filepath)


    def skullstrip(self,fileobj=None,out_file=None,args=None,suffix=None): 

        #setting files
        fileobj, out_file = self.FuncHandler(fileobj,out_file,suffix=suffix)

        args_in = "" #add in terminal flags here (ex: "-overwrite") if you want them called with ubiquity
                    #accross the whole script any time this command is called. Otherwise add flags the the "args" argument of the command
        if args is not None:
            args_in = args_in + args
        #running skull stripping (change this to change skull stripping program)
        fsl.BET(in_file=fileobj,out_file=out_file,args=args_in,output_type="NIFTI_GZ").run()
        #https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.fsl/preprocess.html#bet

        #remove temp files
        if type(fileobj) == models.BIDSImageFile:
            fileobj = os.path.join(self._output_dir,fileobj.filename)
        if "_desc-temp" in fileobj:
            os.remove(fileobj)

    def despike(self,fileobj=None,out_file=None,args=None,suffix=None):

        #setting files
        fileobj, out_file = self.FuncHandler(fileobj,out_file,suffix=suffix)
        args_in = "" #add in terminal flags here (ex: "-overwrite") if you want them called with ubiquity
                    #accross the whole script any time this command is called. Otherwise add flags the the "args" argument of the command
        if args is not None:
            args_in = args_in + args

        afni.Despike(in_file=fileobj,out_file=out_file,args=args_in).run()
        #https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.afni/preprocess.html#despike

        #remove temp files
        if type(fileobj) == models.BIDSImageFile:
            fileobj = os.path.join(self._output_dir,fileobj.filename)
        if "_desc-temp" in fileobj:
            os.remove(fileobj)

    def warp(self,fileobj1=None,fileobj2=None,out_file=None,transformation=None,args=None,saved_mat_file=None,suffix=None):
        """
        https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.afni/preprocess.html#warp

        :param fileobj1:
        :type fileobj1:
        :param fileobj2:
        :type fileobj2:
        :param out_file:
        :type out_file:
        :param transformation:
        :type transformation:
        :param args:
        :type args:
        :param saved_mat_file:
        :type saved_mat_file:
        :param suffix:
        :type suffix:
        :return:
        :rtype:
        """
        #setting files
        if fileobj2 is not None:
            fileobj2, _ = self.FuncHandler(fileobj2,out_file,suffix)
        fileobj1, out_file = self.FuncHandler(fileobj1,out_file,suffix)
    
        ThreeDWarp = afni.Warp(in_file=fileobj1,out_file=out_file)

        if args is not None:
            ThreeDWarp.inputs.args=args
        if transformation == 'card2oblique':
            ThreeDWarp.inputs.oblique_parent = fileobj2
        elif transformation == 'deoblique':
            ThreeDWarp.inputs.deoblique = True
        elif transformation == 'mni2tta':
            ThreeDWarp.inputs.mni2tta = True
        elif transformation == 'tta2mni':
            ThreeDWarp.inputs.tta2mni = True
        elif transformation == 'matrix':
            ThreeDWarp.inputs.matparent = fileobj2
        elif transformation == None:
            print("Warning: no transformation input given")
        else:
            print("Warning: none of the transformation options given match the possible arguments. Matching arguments are card2oblique,"+
             " deoblique, mni2tta, tta2mni, and matrix")
        #ThreeDWarp.inputs.num_threads = cpu_count()

        if saved_mat_file: #this is for if the pipline requires saving the 1D matrix tranformation information
            print('saving matrix')
            ThreeDWarp.inputs.verbose = True
            ThreeDWarp.inputs.save_warp = True
            
        ThreeDWarp.run()

        #remove temp files
        if type(fileobj1) == models.BIDSImageFile:
            fileobj1 = os.path.join(self._output_dir,fileobj1.filename)
        if "_desc-temp" in fileobj1:
            os.remove(fileobj1)

    def axialize(self,fileobj=None,out_file=None,args=None,suffix=None):

        fileobj, out_file = self.FuncHandler(fileobj,out_file,suffix=suffix)

        args_in = "-overwrite" #add in terminal flags here (ex: "-overwrite") if you want them called with ubiquity
                    #accross the whole script any time this command is called. Otherwise add flags the the "args" argument of the command
        if args is not None:
            args_in = args_in + args
        
        afni.Axialize(in_file=fileobj,out_file=out_file, args=args_in).run()
        #https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.afni/utils.html#axialize

        #remove temp files
        if type(fileobj) == models.BIDSImageFile:
            fileobj = os.path.join(self._output_dir,fileobj.filename)
        if "_desc-temp" in fileobj:
            os.remove(fileobj)

    def volreg(self,in_file,out_file=None,suffix=None,base=None,tshift=False,interpolation="heptic",onedfile=None,onedmat=None):

        in_file, out_file = self.FuncHandler(in_file,out_file,suffix=suffix)
        myreg = afni.Volreg(in_file=in_file,out_file=out_file)
        #https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.afni/preprocess.html#volreg

        if base is not None:
            base, _ = self.FuncHandler(base,out_file,suffix)
            myreg.inputs.basefile = base

        myreg.inputs.verbose = self.is_verbose
        myreg.inputs.timeshift = tshift
        myreg.inputs.interp = interpolation
        if onedfile is True:
            myreg.inputs.oned_file = out_file.replace(".nii.gz",".1D")
        elif onedfile is not None:
            myreg.inputs.oned_file = onedfile
        if onedmat is True:
            myreg.inputs.oned_matrix_save = out_file.replace("vrA.nii.gz","vrmat.aff12.1D")
        elif onedmat is not None:
            myreg.inputs.oned_matrix_save = onedmat
        myreg.inputs.num_threads = cpu_count() #should improve speed

        myreg.run()

        #remove temp files
        if type(in_file) == models.BIDSImageFile:
            in_file = os.path.join(self._output_dir,in_file.filename)
        if "_desc-temp" in in_file:
            os.remove(in_file)

    def copy(self,in_file,out_file=None,suffix=None):

        if out_file is None and suffix is None:
            suffix = "copy"
        in_file, out_file = self.FuncHandler(in_file,out_file,suffix)
        copy3d = afni.Copy()
        #https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.afni/utils.html#copy

        copy3d.inputs.in_file = in_file
        copy3d.inputs.out_file = out_file
        copy3d.inputs.verbose = self.is_verbose
        #copy3d.inputs.num_threads = cpu_count()
        copy3d.run()

        #remove temp files
        if type(in_file) == models.BIDSImageFile:
            in_file = os.path.join(self._output_dir,in_file.filename)
        if "_desc-temp" in in_file:
            os.remove(in_file)

    def onedcat(self,in_files,out_file,sel=None): 

        if type(in_files) is str:
            in_files = [in_files]
        
        cat = afni.Cat(in_files=in_files,out_file=out_file)
        #https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.afni/utils.html#cat

        if sel is not None:
            if not "'" in sel and type(sel) is str:
                sel="'{string}'".format(string=sel)
            cat.inputs.sel = sel
        cat.run()

    def tshift(self,in_file=None,out_file=None, suffix=None, interp="heptic"):
        
        mycwd = os.getcwd()
        os.chdir(self._output_dir)
        in_file, out_file = self.FuncHandler(in_file,out_file,suffix)
        #out_file = out_file.replace(".nii.gz","")
        mytshift = afni.TShift(in_file=in_file, interp=interp,outputtype='NIFTI_GZ')
        #https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.afni/preprocess.html#tshift

        #mytshift.inputs.args = "-prefix %s & %s " % (out_file, out_file.) 
        mytshift.run()
        os.rename(in_file.replace(".nii.gz","_tshift.nii.gz"),out_file)
        os.chdir(mycwd)

        self.refit(in_file=out_file,args="-view orig")

        #remove temp files
        if type(in_file) == models.BIDSImageFile:
            in_file = os.path.join(self._output_dir,in_file.filename)
        if "_desc-temp" in in_file:
            os.remove(in_file)

    def refit(self,in_file,out_file=None,deoblique=False,args=""):

        myfit = afni.Refit(in_file=in_file)
        #https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.afni/utils.html#refit

        myfit.inputs.deoblique = deoblique
        myfit.inputs.args = args
        myfit.run()

        #remove temp files
        if type(in_file) == models.BIDSImageFile:
            in_file = os.path.join(self._output_dir,in_file.filename)
        if "_desc-temp" in in_file:
            os.remove(in_file)

    def allineate(self,in_file,out_file=None,suffix=None,final="nearestneighbor",mat=None,base=None,args=""):

        in_file, out_file = self.FuncHandler(in_file,out_file,suffix)

        myalline = afni.Allineate(in_file=in_file,out_file=out_file)
        #https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.afni/preprocess.html#allineate

        myalline.inputs.args = args
        #myalline.inputs.num_threads = cpu_count()
        if mat is not None:
            mat , _ = self.FuncHandler(mat,out_file,suffix)
            myalline.inputs.in_matrix = mat
        if base is not None:
            myalline.inputs.reference = base
        myalline.run()

        #remove temp files
        if type(in_file) == models.BIDSImageFile:
            in_file = os.path.join(self._output_dir,in_file.filename)
        if "_desc-temp" in in_file:
            os.remove(in_file)

    def zcat(self,in_files,out_file,suffix=None,args=""):

        data = []
        for in_file in in_files:
            x, _ = self.FuncHandler(in_file,"_",suffix)
            data.append(x)

        myzcat = afni.Zcat(in_files=data,out_file=out_file)
        #https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.afni/utils.html#zcat

        myzcat.inputs.args = args
        myzcat.run()

        if type(in_file) == models.BIDSImageFile:
            in_file = os.path.join(self._output_dir,in_file.filename)
        if "_desc-temp" in in_file:
            os.remove(in_file)

    def calc(self,in_file_a,expr,in_file_b=None,in_file_c=None,output=None,suffix=None):

        in_file_a, output = self.FuncHandler(in_file_a,output,suffix)
        in_file_b, _ = self.FuncHandler(in_file_b,output,suffix)
        in_file_c, _ = self.FuncHandler(in_file_c,output,suffix)

        mycalc = afni.Calc(in_file_a=in_file_a,in_file_b=in_file_b,expr=expr,out_file=output)
        #https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.afni/utils.html#calc

        mycalc.inputs.outputtype = "NIFTI_GZ"
        mycalc.inputs.num_threads = cpu_count()
        mycalc.run()

        for in_file in [in_file_a,in_file_b]:
            if type(in_file) == models.BIDSImageFile:
                in_file = os.path.join(self._output_dir,in_file.filename)
            if "_desc-temp" in in_file:
                os.remove(in_file)

    @property
    def is_verbose(self):
        return self.is_verbose

    @is_verbose.setter
    def is_verbose(self, value):
        self._is_verbose = value


if __name__ == "__main__":
    args = get_parser().parse_args()
    pre = Preprocessing(**vars(args))

    echo_times = [12.3,26,40]

    #delete any preprocessing files not supposed to be there
    for root,_,files in os.walk(pre._output_dir):
        for file in files:
            if ".json" not in file:
                filepath = os.path.join(root,file)
                os.remove(filepath)

    #getting all the subjects into place
    sub_ids = pre.BIDS_layout.get_subjects()

    #Main preprocessing pipeline: uses tools defined above
    for sub_id in sub_ids :

        #Defining which images we care about and setting the basenames
        all_fobj = []
        for BIDSFiles in pre.BIDS_layout.get(scope='raw',subject=sub_id,suffix='bold',extension='.nii.gz'):
            all_fobj.append(BIDSFiles)

        for BIDSFiles in pre.BIDS_layout.get(scope='raw',extension='.nii.gz',suffix='T1w',acquisition='MPRAGE', subject=sub_id):
            all_fobj.append(BIDSFiles)

        #copying those files to a new derivatives directory so we can mess with them
        for fobj in all_fobj:
            if pre.is_verbose:
                print("copying {filename} to preprocessing derivatives directory".format(filename=fobj.path))
            try:
                copy(fobj.path,os.path.join(pre._output_dir,fobj.filename))
            except SameFileError:
                print("{files} already exists in the preprocessing directory, overwriting...".format(files=fobj.filename))

        #set derivatives directory
        if not sub_id is sub_ids[0]:
            pre.set_bids(vars(args)['include'],vars(args)['exclude'])
        pre.BIDS_layout.add_derivatives(pre._output_dir)

        # skull stripping
        if pre.is_verbose:
            print("performing skull stripping")
        filenames = pre.BIDS_layout.get(scope='derivatives', subject=sub_id, extension='.nii.gz',return_type="filename",acquisition="MPRAGE")
        for filename in filenames:
            pre.skullstrip(filename)
            pre.warp(filename,transformation='deoblique',out_file=filename.split('.nii.gz')[0] + "_do.nii.gz")
        
        #Calculate and save motion and obliquity parameters, despiking first if not disabled, and separately save and mask the base volume
        #assuming only one anatomical image
        
        for run in pre.BIDS_layout.get(scope='derivatives', subject=sub_id, extension='.nii.gz', suffix='bold',target='run',return_type='id'):

            CWD = os.getcwd()
            os.chdir(pre._output_dir)
            if pre.is_verbose:
                print("denoising and saving motion parameters")
            for fobj in pre.BIDS_layout.get(scope='derivatives', subject=sub_id, extension='.nii.gz', suffix='bold',run=run):
                try: #checking that image is either single echo or the first of the multi echo series
                    if not fobj.get_entities()['echo'] == "01":
                        continue
                except KeyError: #handles single echo
                    pass
                for anat in pre.BIDS_layout.get(scope='derivatives', subject=sub_id, extension='.nii.gz',acquisition='MPRAGE'):
                    pre.warp(fileobj2=fobj.path,fileobj1=anat.path,args="-newgrid 1.000000",saved_mat_file=True, transformation='card2oblique',
                    suffix="anat_to_s"+str(fobj.get_entities()['subject']).zfill(2)+"r"+str(fobj.get_entities()['run']).zfill(2))  #saving the transformation matrix for later
                no_echo = re.sub(r"echo-[0-9]{2}_bold","bold",fobj.path.replace(".nii.gz","_desc-vrA.nii.gz"))
                pre.despike(fobj.path,out_file=no_echo)
                pre.axialize(no_echo)
                pre.copy(no_echo + "[2]",no_echo.replace("_desc-vrA.nii.gz","_eBase.nii.gz"))
                pre.volreg(no_echo,base=no_echo.replace("_desc-vrA.nii.gz","_eBase.nii.gz"),interpolation='heptic',
                    onedfile=True,onedmat=no_echo.replace("vrA.nii.gz","vrmat.aff12.1D"))
                pre.onedcat(no_echo.replace(".nii.gz",".1D"),os.path.join(fobj.dirname,"sub-{sub}_run-{run}_motion.1D".format(
                    sub=str(fobj.get_entities()['subject']).zfill(2),run=str(fobj.get_entities()['run']).zfill(2))),"[0..5]{1..$}")

            
            #
            if pre.is_verbose:
                print("Starting preprocessing of functional datasets")
            for fobj in pre.BIDS_layout.get(scope='derivatives', subject=sub_id, extension='.nii.gz', suffix='bold',run=run):
                pre.despike(fobj.path, fobj.path.replace(".nii.gz","_desc-pt.nii.gz"))
                pre.tshift(fobj.path.replace(".nii.gz","_desc-pt.nii.gz"), fobj.path.replace(".nii.gz","_desc-ts.nii.gz"),interp="heptic")
                pre.axialize(fobj.path.replace(".nii.gz","_desc-ts.nii.gz"))
                pre.refit(fobj.path.replace(".nii.gz","_desc-ts.nii.gz"),deoblique=True,args="-TR 2.5")
            #print("Performing cortical reconstruction on %s" %sub_id)
            #preprocess.cortical_recon(bids_obj)

            if pre.is_verbose:
                print("Prepare T2* and S0 volumes for use in functional masking and (optionally) anatomical-functional coregistration (takes a little while).")
            fobjs = []
            for fobj in pre.BIDS_layout.get(scope='derivatives', subject=sub_id, extension='.nii.gz', suffix='bold',run=run):
                no_echo = re.sub(r"echo-[0-9]{2}_bold","bold",fobj.path.replace(".nii.gz","_desc-vrA.nii.gz"))
                try:
                    assert fobj.get_entities()['echo'] 
                    echo = str(fobj.get_entities()['echo']).zfill(2)
                except (AssertionError, KeyError): #handled in case of single echo BOLD data
                    echo = "00"
                pre.allineate(fobj.path.replace(".nii.gz","_desc-ts.nii.gz[2..22]"),fobj.path.replace(".nii.gz","_desc-e{s}_vrA.nii.gz".format(s=echo)),
                    mat=no_echo.replace("vrA.nii.gz","vrmat.aff12.1D{2..22}"),base=no_echo.replace("_desc-vrA.nii.gz","_eBase.nii.gz"),
                    args="-final NN -NN -float")
                fobjs.append(fobj.path.replace(".nii.gz","_desc-e{s}_vrA.nii.gz".format(s=echo)))
            newname = os.path.join(fobj.dirname, "run-{r}_basestack.nii.gz".format(r=run))
            if len(fobjs) >= 2: #multi echo
                pre.zcat(fobjs,newname)
                t2smap_workflow(newname,echo_times)#,label="run-{r}_t2".format(r=run))
            elif len(fobjs) == 1: #single echo
                newname = fobjs[0]
            else:
                raise FileNotFoundError

            if pre.is_verbose:
                print("--------Using AFNI align_epi_anat.py to drive anatomical-functional coregistration ")
            #pre.calc()
            
            os.chdir(CWD)
    #if pre._is_verbose:
    #    tree(pre.BIDS_layout.root)