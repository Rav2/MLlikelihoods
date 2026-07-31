#!/usr/bin/env python

import glob, time

def clean ( line : str ) -> str:
    return line.replace("+","")

def collect ( case : str = "Sleptons" ) -> int:
    path = f"tables/*/*-{case}_*.csv"
    files = glob.glob ( path )
    sm_case = "slep" if case == "Sleptons" else "chiwzoff"
    sm_path = f"../smodels-utils/stats_ml/full_{sm_case}/T*.csv"
    files += glob.glob ( sm_path )
    outfile = f"{case}.csv"
    first = True
    ct_l = 0
    with open ( outfile, "wt" ) as out:
        for fname in files:
            with open ( fname, "rt" ) as f:
                lines = f.readlines()
                #if len(lines)>1:
                #    print ( f"{fname}: {len(lines)-1} lines" )
                if first:
                    out.write ( clean ( lines[0] ) )
                    first = False
                for line in lines[1:]:
                    out.write ( clean ( line ) )
                    ct_l += 1
        out.close()
        print ( f"{outfile}: {ct_l} lines" )
    return ct_l

def log( n_lines : dict ):
    with open ( f"collection.log", "wt" ) as f:
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

# tables/1911.12606-1-0/table-Sleptons_bkgonly-Sleptons_patchset-0.csv
# tables/1911.12606-10-0/results-EWKinos_bkgonly-EWKinos_patchset-0.csv

if __name__ == "__main__":
    collectAll()
