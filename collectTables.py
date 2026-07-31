#!/usr/bin/env python

import glob

def clean ( line : str ) -> str:
    return line.replace("+","")

def collect ( case : str = "Sleptons" ):
    path = f"tables/*/*-{case}_*.csv"
    files = glob.glob ( path )
    sm_case = "slep" if case == "Sleptons" else "chiwzoff"
    sm_path = f"../smodels-utils/stats_ml/full_{sm_case}/T*.csv"
    files += glob.glob ( sm_path )
    outfile = f"{case}.csv"
    first = True
    with open ( outfile, "wt" ) as out:
        for fname in files:
            with open ( fname, "rt" ) as f:
                lines = f.readlines()
                if len(lines)>1:
                    print ( f"{fname}: {len(lines)-1} lines" )
                if first:
                    out.write ( clean ( lines[0] ) )
                    first = False
                for line in lines[1:]:
                    out.write ( clean ( line ) )
        out.close()

def collectAll():
    cases = [ "Sleptons", "EWKinos" ]
    for case in cases:
        collect ( case )

# tables/1911.12606-1-0/table-Sleptons_bkgonly-Sleptons_patchset-0.csv
# tables/1911.12606-10-0/results-EWKinos_bkgonly-EWKinos_patchset-0.csv

if __name__ == "__main__":
    collectAll()
