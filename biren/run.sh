if [ "$1" == "compile" ]; then
    brcc -o test_spmm_time test_spmm_time.cpp -x supa
    # brcc test_sddmm_time.cpp -o test_sddmm_time -x supa
fi

if [ "$1" == "spmm" ]; then
    dir=/home/lin/biren/biren/biren/data/mm
    for((nv=8;nv<=512;nv*=2));
    do
        for i in ${dir}/*;
        do
            file="${i}"
            file=${file##*/}
            
            echo "${i}/${file}.mtx"
            ./test_spmm_time "${i}/${file}.mtx" "${nv}" >> "../result/spmm_result_${nv}.csv"
        done
    done
fi

if [ "$1" == "sddmm" ]; then
    dir=/home/lin/biren/biren/biren/data/mm
    for i in ${dir}/*;
    do
        file="${i}"
        file=${file##*/}

        if [ "$file" == "cage15" | "$file" == "case39"]; then
            echo "${i}/${file}.mtx passed"
        else
            echo "${i}/${file}.mtx"
            # SUPA_VISIBLE_DEVICES=1 ./test_sddmm_time ../data/mm/a5esindl/a5esindl.mtx 128
            ./test_sddmm_time "${i}/${file}.mtx" 128 >> "../result/sddmm_result_128.csv"
        fi

    done
fi

if [ "$1" == "total" ]; then
    dir=/home/lin/biren/biren/biren/data/mm
    for((nv=128;nv<=512;nv*=2));
    do
        for i in ${dir}/*;
        do
            file="${i}"
            file=${file##*/}
            
            echo "${i}/${file}.mtx"
            ./test_spmm_time "${i}/${file}.mtx" "${nv}" >> "../result/spmm_result_${nv}.csv"
        done
    done
    for i in ${dir}/*;
    do
        file="${i}"
        file=${file##*/}

        if [ "$file" == "cage15" | "$file" == "case39"]; then
            echo "${i}/${file}.mtx passed"
        else
            echo "${i}/${file}.mtx"
            # SUPA_VISIBLE_DEVICES=1 ./test_sddmm_time ../data/mm/a5esindl/a5esindl.mtx 128
            ./test_sddmm_time "${i}/${file}.mtx" 128 >> "../result/sddmm_result_128.csv"
        fi
    done
fi