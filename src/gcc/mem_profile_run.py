import subprocess, sqlite3

#valgrind -q --tool=massif --massif-out-file=mem_profile.massif --depth=1 ./mem_profilee 2 8 S H
##ms_print massif.out.`cat mem_profile_pid.txt`
##ms_print mem_profile.massif
#grep mem_heap_B mem_profile.massif |cut -c 12- |sort -n | tail -n 1

def get_usage(d,m, use_compiled, use_lyndon):
    args = ["valgrind", "-q", "--tool=massif", "--massif-out-file=mem_profile.massif",
            "--depth=1", "./mem_profilee",
            str(d), str(m),
            "C" if use_compiled else "S",
            "L" if use_lyndon else "H",
            ]
    b=subprocess.check_output(args)
    extra_memory = int(b) if (use_compiled and d>0) else 0
    a = subprocess.check_output(
        "grep mem_heap_B mem_profile.massif |cut -c 12- |sort -n | tail -n 1",
        shell=True)
    return int(a)+extra_memory

con = sqlite3.connect("mem_profile.sqlite")
con.cursor().execute("create table if not exists A (METHOD TEXT, BASIS TEXT, D INT, M INT, BYTES INT, PERFORMED TEXT, VERSION TEXT)")
con.commit()
version=1

def test(d,m, use_compiled, use_lyndon, no_commit):
    usage=get_usage(d,m, use_compiled, use_lyndon)
    method = "C" if use_compiled else "S"
    basis = "L" if use_lyndon else "H"
    print(method, basis, d,m,usage)
    con.cursor().execute("INSERT INTO A(METHOD,BASIS,D,M,BYTES,PERFORMED,VERSION) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)",
                         (method, basis, d,m,usage,version))
    if not no_commit:
        con.commit()

no_commit=0
if no_commit:
    print("warning: not saving results")
        
#test(0,0,True,1,no_commit)
for use_lyndon in [True]:#,False]:
    for use_compiled in [True,False]:
        for d in [3]:
            for m in range(10,11):
                test(d,m, use_compiled, use_lyndon, no_commit)
    
