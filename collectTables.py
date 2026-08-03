#!/usr/bin/env python

import glob, time, shutil, os, subprocess

def clean ( line : str ) -> str:
    return line.replace("+","")

def collect ( case : str = "Sleptons" ) -> int:
    path = f"tables/*/*-{case}_*.csv"
    files = glob.glob ( path )
    sm_case = "slep" if case == "Sleptons" else "chiwzoff"
    sm_path = f"../smodels-utils/stats_ml/full_{sm_case}/T*.csv"
    files_sms = glob.glob ( sm_path )
    outfile = f"{case}.csv"
    first = True
    ct_l = 0
    with open ( outfile, "wt" ) as out:
        for fname in files:
            with open ( fname, "rt" ) as f:
                lines = f.readlines()
                if first:
                    out.write ( clean ( lines[0] ) )
                    first = False
                for line in lines[1:]:
                    tokens = line.split(",")
                    out.write ( clean ( line ) )
                    ct_l += 1
        for fname in files_sms:
            with open ( fname, "rt" ) as f:
                lines = f.readlines()
                for line in lines[1:-1]:
                    tokens = line.split(",")
                    out.write ( clean ( line ) )
                    ct_l += 1
        out.close()
        print ( f"{outfile}: {ct_l} lines" )
    return ct_l

def log( n_lines : dict ):
    logfile = "collection.log"
    if os.path.exists ( logfile ):
        backup = "collection.backup"
        shutil.copy ( logfile, backup )
    with open ( logfile, "wt" ) as f:
        f.write ( "{\n" )
        f.write ( f"    'time': {time.asctime()},\n" )
        for k, v in n_lines.items():
            f.write ( f"    '{k}': {v},\n" )
        f.write ( "}\n" )

def collectAll():
    cases = [ "Sleptons", "EWKinos" ]
    n_lines = {}
    for case in cases:
        nl  = collect ( case )
        n_lines[case]=nl
    log ( n_lines )
    
def postprocess():
    cmd = "cp EWKinos.csv Sleptons.csv ~/git/likelihoods/data/"
    o = subprocess.getoutput ( cmd )
    print ( f"{cmd}: {o}" )
    cmd = "cd ~/git/likelihoods/data && python ./conv_csv_npy.py"
    o = subprocess.getoutput ( cmd )
    print ( f"{cmd}: {o}" )

# tables/1911.12606-1-0/table-Sleptons_bkgonly-Sleptons_patchset-0.csv
# tables/1911.12606-10-0/results-EWKinos_bkgonly-EWKinos_patchset-0.csv

if __name__ == "__main__":
    collectAll()
    postprocess()
