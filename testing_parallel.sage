import time
import cython 
load("utils.sage")
load("bridge_to_c.pyx")
import multiprocessing 
ncpus = multiprocessing.cpu_count()
print(f"{ncpus} CPUs available")

def get_codeword_with_fixed_hw(H, hw, starting_vector = 0):
    '''
    returns the first codeword with fixed hammingweight that comes lexicographically after starting_vector

    H: the public key of a McEliece instance
    hw: Hammingweight of desired vector
    '''
    n = H.ncols()
    k = H.nrows()
    if starting_vector == 0:
        current_vector = 2**hw - 1
    else:
        current_vector = sage_next_subset(starting_vector, 1 << n)
    VS = VectorSpace(GF(2), n)
    while True:
        vec = VS(Integer(current_vector).digits(base=2, padto=n))
        if (H * vec.column()) == zero_vector(GF(2), k).column():
            indices_of_ones = []
            for index in range(len(vec)):
                if vec[index] == 1:
                    indices_of_ones.append(ZZ(index))
            return vec, indices_of_ones, current_vector
        else:
            current_vector = sage_next_subset(current_vector, 1 << n)

def read_H_f_from_file(path, filename, n, k):
    H = matrix(GF(2), n-k, n)
    flist = []
    i = 0
    with open(path+filename, 'r') as f:
        for line in f:
            if i<n-k:
                line = line.strip('[').strip(']\n')
                H[i] = vector([GF(2)(el) for el in line.split(' ')])
            else:
                line = line.strip('[').strip(']\n')
                flist = [GF(2)(el) for el in line.split(', ')]
            i+=1

    return H, flist

def read_L_g_from_file(filename, ff):
    with open(filename, 'r') as f:
        i = 0
        for line in f:
            if i == 0:
                line = line.strip('[').strip(']\n')
                glist = [ff(el) for el in line.split(', ')]
            elif i==1:
                line = line.strip('[').strip(']\n')
                L = vector([ff(el) for el in line.split(', ')])
            else:
                raise ValueError('inconsistend data in file', filename)
            i+=1
    return L, glist

def bitstring(length, nr_ones, index):
    '''
    returns the bitstring from the sorted set of bitstrings of given length and given hammingweight with given index
    '''
    if length == 0:
        return []
    elif index < binomial(length - 1, nr_ones):
        return [0] + bitstring(length - 1, nr_ones, index)
    else:
        return [1] + bitstring(length - 1, nr_ones - 1, index - binomial(length - 1, nr_ones))
    
def divide_searchspace(length, nr_ones):
    '''
    returns the starting-vectors for parallelizing searching all bitstrings of fixed length and hammingweight
    '''
    nr_bitstrings = binomial(length, nr_ones)
    interval = nr_bitstrings // ncpus
    starting_indices = [i * interval for i in range(ncpus)]
    return starting_indices
    
def test_function_parallel_search(H, hw, n, starting_vector, end_vector):
    '''
    returns the vector of given hammingweigth that can recover the most positions
    '''
    nrows = H.nrows()
    ncols = H.ncols()
    flag_complete_break = False
    current_vector = int(Integer(starting_vector, base=2))
    end_vector = int(Integer(end_vector, base=2))
    VS = VectorSpace(GF(2), n)
    output = []
    best_performing_vector = ["", "", []]
    while current_vector != 0 and bin(current_vector).count('1') == hw and current_vector <= end_vector:
        # get current vector
        vec = VS(Integer(current_vector).digits(base=2, padto=n))
        # check whether the vector is a codeword
        if (H * vec.column()) == zero_vector(GF(2), nrows).column():
            # get support of vector
            support = get_support_of_vector(vec)
            
            Hpub_abr_ = H[[i for i in range(nrows)], [j for j in range(ncols) if j in support]]
            if Hpub_abr_.is_invertible():
                flag_complete_break = True
                print(f"COMPLETE BREAK POSSIBLE:\nCODEWORD: {vec}\nSUPPORT: {support}")
                output = [[vec, support, [i for i in range(nrows) if not (i in support)]]]
                best_performing_vector = [vec, support, [i for i in range(nrows) if not (i in support)]]
                return 1, output, best_performing_vector
            
            # checking what can be recovered
            recoverable = []
            recoverable_in_current_iteration = []
            for r in range(nrows):
                if r in support:
                    continue
                J = support + recoverable
                J.append(r)
                H_prime = H[[i for i in range(nrows)], [j for j in range(ncols) if j in J]]
                if Hpub_abr_.rank() == H_prime.rank():
                    recoverable.append(r)
                    recoverable_in_current_iteration.append(r)

            while recoverable_in_current_iteration != []:
                recoverable_in_current_iteration = []
                for r in range(nrows):
                    if r in support or r in recoverable:
                        continue
                    J = support + recoverable
                    J.append(r)
                    H_prime = H[[i for i in range(nrows)], [j for j in range(ncols) if j in J]]
                if Hpub_abr_.rank() == H_prime.rank():
                    recoverable.append(r)
                    recoverable_in_current_iteration.append(r) 
            
            # print 
            if recoverable != []:
                #print(f"CODEWORD: {vec}\nSUPPORT: {support}\nRECOVERABLE: {recoverable}\n\n")
                if len(recoverable) > len(best_performing_vector[2]):
                    # update best performing vector
                    best_performing_vector = [vec, support, recoverable]
                output.append([vec, support, recoverable])
        
        current_vector = sage_next_subset(current_vector, 1 << n)
    return 0, output, best_performing_vector


McEliece0 = {'label': 0, 'n': 41,  'k': 11, 'w': 5, 'm': 6, 'f': x^6 + x + 1}
McEliece1 = {'label': 1, 'n': 3488, 'k': 2720, 'w': 64, 'm': 12, 'f':x^12+x^3+1}
McEliece2 = {'label': 2,'n': 4608, 'k': 3360, 'w': 96, 'm': 13, 'f':x^13+x^4+x^3+x+1}
McEliece3 = {'label': 3,'n': 6960, 'k': 5413, 'w': 119,'m': 13, 'f':x^13+x^4+x^3+x+1}
McEliece4 = {'label': 4,'n': 8192, 'k': 6528, 'w': 128,'m': 13, 'f':x^13+x^4+x^3+x+1}
McEliece22 = {'label': 22, 'n': 32, 'k': 22, 'w': 2, 'm': 5, "bitcomplexity": 22.0, 'f':x^5+x^2+1}
McEliece39 = {'label': 39, 'n': 28, 'k': 18, 'w': 2, 'm': 5, "bitcomplexity": 39.0, 'f':x^5+x^2+1}
#McEliece
param_set = McEliece22
print("McEliece 22")
#param_set = McEliece39
#print("McEliece 39")

R = ZZ['x']
p = 2
kappa = param_set['m']
n_max = p**kappa - 1
F = GF(p)
Fx = PolynomialRing(F, 'x')
defining_poly = param_set['f']
print('defining_poly:', defining_poly)
ff.<a> = FiniteField(p**kappa, modulus = defining_poly)
R = PolynomialRing(ff, 'x')
x = R.gen()

n = param_set['n']
k = param_set['k']
t = param_set['w']
H, _ = read_H_f_from_file("FILEPATH", "", n, k)
L, g = read_L_g_from_file("FILEPATH", ff)

def get_support_of_vector(vec):
    support = []
    for index in range(len(vec)):
        if vec[index] != 0:
            support.append(ZZ(index))
    return support

result_list = []

def parallel():
    def log_result(result):
        global result_list
        if result[0] == 0:
            result_list.append(result)
        elif result[0] == 1:
            result_list = [result]
            pool.terminate()
    global result_list
    result_list = []
    for hw in range(2*t+1, 12):
        starting_indices = divide_searchspace(n, hw)
        starting_vector = [bitstring(n, hw, item) for item in starting_indices]
        for item in starting_vector:
            item.reverse()
        starting_vector.append((n-hw)*[0]+[1]*hw)
        
        pool = multiprocessing.Pool()
        print(f"Hammingweight {hw}")
        for i in range(len(starting_vector)-1):
            res = pool.apply_async(test_function_parallel_search, args = (H, hw, n, starting_vector[i], starting_vector[i+1]), callback = log_result)
        pool.close()
        pool.join()
        best_performing_vectors = [item[2] for item in result_list]
        maximum = []
        max_item = []
        for item in best_performing_vectors:
            if len(item[2]) > len(maximum):
                maximum = item[2]
                max_item = item
        if max_item != []:
            print("BEST PERFORMING VECTOR: ")
            print(f"CODEWORD: {max_item[0]}\nSUPPORT: {max_item[1]}\nRECOVERABLE: {max_item[2]}\n\n")
        else:
            print(f"NO RECOVERY POSSIBLE WITH HW {hw}")

def main():
    start = time.perf_counter()
    multiprocessing.set_start_method('fork')
    parallel()
    end = time.perf_counter()
    print(f"TIME NEEDED: {end-start}")


if __name__ == "__main__":
    main()
