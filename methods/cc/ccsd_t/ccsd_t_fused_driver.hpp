
#ifndef CCSD_T_FUSED_HPP_
#define CCSD_T_FUSED_HPP_

#include "ccsd_t_all_fused.hpp"
#include "ccsd_t_common.hpp"

int check_device(long);
int device_init(long iDevice,int *gpu_device_number );
void dev_release();
void finalizememmodule();

//
//
//
template<typename T>
std::tuple<double,double,double,double>
ccsd_t_fused_driver_new(SystemData& sys_data, ExecutionContext& ec,
                        std::vector<int>& k_spin,
                        const TiledIndexSpace& MO,
                        Tensor<T>& d_t1, Tensor<T>& d_t2,
                        Tensor<T>& d_v2,
                        std::vector<T>& k_evl_sorted,
                        double hf_ccsd_energy, int iDevice,
                        bool is_restricted,
                        LRUCache<Index,std::vector<T>>& cache_s1t, LRUCache<Index,std::vector<T>>& cache_s1v,
                        LRUCache<Index,std::vector<T>>& cache_d1t, LRUCache<Index,std::vector<T>>& cache_d1v,
                        LRUCache<Index,std::vector<T>>& cache_d2t, LRUCache<Index,std::vector<T>>& cache_d2v,
                        bool seq_h3b=false, bool tilesize_opt=true)
{
  //
  auto rank     = ec.pg().rank().value();
  bool nodezero = rank==0;

  Index noab=MO("occ").num_tiles();
  Index nvab=MO("virt").num_tiles();

  Index noa=MO("occ_alpha").num_tiles();
  Index nva=MO("virt_alpha").num_tiles();

  auto mo_tiles = MO.input_tile_sizes();
  std::vector<size_t> k_range;
  std::vector<size_t> k_offset;
  size_t sum = 0;
  for (auto x: mo_tiles){
    k_range.push_back(x);
    k_offset.push_back(sum);
    sum+=x;
  }

  if(nodezero){
    cout << "noa,nva = " << noa << ", " << nva << endl;
    cout << "noab,nvab = " << noab << ", " << nvab << endl;
    // cout << "k_spin = " << k_spin << endl;
    // cout << "k_range = " << k_range << endl;
    cout << "MO Tiles = " << mo_tiles << endl;
  }

  //Check if node has number of devices specified in input file
  int dev_count_check;
#if defined(USE_CUDA)
  cudaGetDeviceCount(&dev_count_check);
  if(dev_count_check < iDevice){
    if(nodezero) cout << "ERROR: Please check whether you have " << iDevice <<
      " cuda devices per node. Terminating program..." << endl << endl;
    return std::make_tuple(-999,-999,0,0);
  }
#elif defined(USE_HIP)
  hipGetDeviceCount(&dev_count_check);
  if(dev_count_check < iDevice){
    if(nodezero) cout << "ERROR: Please check whether you have " << iDevice <<
      " hip devices per node. Terminating program..." << endl << endl;
    return std::make_tuple(-999,-999,0,0);
  }
#elif defined(USE_DPCPP)
  {
    cl::sycl::gpu_selector device_selector;
    cl::sycl::platform platform(device_selector);
    auto const& gpu_devices = platform.get_devices();
    for (auto &gpu_device : gpu_devices) {
      const std::string deviceName = gpu_device.get_info<cl::sycl::info::device::name>();
      if (gpu_device.is_gpu() && (deviceName.find("Intel") != std::string::npos))
        dev_count_check++;
    }
    if(dev_count_check < iDevice) {
      if(nodezero) cout << "ERROR: Please check whether you have " << iDevice <<
                     " SYCL devices per node. Terminating program..." << endl << endl;
      return std::make_tuple(-999,-999,0,0);
    }
    else if (dev_count_check <= 0) {
      if(nodezero) cout << "ERROR: NO SYCL devices found on node, " <<
                     "Terminating program..." << endl << endl;
      return std::make_tuple(-999,-999,0,0);
    }
    else if (dev_count_check > 1) {
      if(nodezero) cout << "ERROR: TODO multi-device per node support for SYCL is not yet supported. "
                        << " Terminating program..." << endl << endl;
      return std::make_tuple(-999,-999,0,0);
    }
  }
#endif

  int gpu_device_number=0;
  //Check whether this process is associated with a GPU
  auto has_GPU = check_device(iDevice);

  // printf ("[%s] rank: %d, has_GPU: %d, iDevice: %d\n", __func__, rank, has_GPU, iDevice);

  if(iDevice==0) has_GPU=0;
  // cout << "rank,has_gpu" << rank << "," << has_GPU << endl;
  if(has_GPU == 1){
    device_init(iDevice, &gpu_device_number);
    // if(gpu_device_number==30) // QUIT
  }
  if(nodezero) std::cout << "Using " << iDevice << " gpu devices per node" << endl << endl;
  //std::cout << std::flush;

  //TODO replicate d_t1 L84-89 ccsd_t_gpu.F

  double energy1 = 0.0;
  double energy2 = 0.0;
  std::vector<double> energy_l(2);
  std::vector<double> tmp_energy_l(2);

  AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
  ac->allocate(0);
  int64_t taskcount = 0;
  int64_t next = ac->fetch_add(0, 1);

  auto cc_t1 = std::chrono::high_resolution_clock::now();

  size_t max_pdim = 0;
  size_t max_hdim = 0;
  for (size_t t_p4b=noab; t_p4b < noab + nvab; t_p4b++)
    max_pdim = std::max(max_pdim,k_range[t_p4b]);
  for (size_t t_h1b = 0; t_h1b < noab; t_h1b++)
    max_hdim = std::max(max_hdim,k_range[t_h1b]);

  size_t max_d1_kernels_pertask = 9*noab;
  size_t max_d2_kernels_pertask = 9*nvab;
  if(tilesize_opt) {
    max_d1_kernels_pertask = 9*noa;
    max_d2_kernels_pertask = 9*nva;
  }

  //
  size_t size_T_s1_t1 = 9 * (max_pdim) * (max_hdim);
  size_t size_T_s1_v2 = 9 * (max_pdim * max_pdim) * (max_hdim * max_hdim);
  size_t size_T_d1_t2 = max_d1_kernels_pertask * (max_pdim * max_pdim) * (max_hdim * max_hdim);
  size_t size_T_d1_v2 = max_d1_kernels_pertask * (max_pdim) * (max_hdim * max_hdim * max_hdim);
  size_t size_T_d2_t2 = max_d2_kernels_pertask * (max_pdim * max_pdim) * (max_hdim * max_hdim);
  size_t size_T_d2_v2 = max_d2_kernels_pertask * (max_pdim * max_pdim * max_pdim) * (max_hdim);

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DCPP)
  T* df_host_pinned_s1_t1 = (T*)getHostMem(sizeof(double) * size_T_s1_t1);
  T* df_host_pinned_s1_v2 = (T*)getHostMem(sizeof(double) * size_T_s1_v2);
  T* df_host_pinned_d1_t2 = (T*)getHostMem(sizeof(double) * size_T_d1_t2);
  T* df_host_pinned_d1_v2 = (T*)getHostMem(sizeof(double) * size_T_d1_v2);
  T* df_host_pinned_d2_t2 = (T*)getHostMem(sizeof(double) * size_T_d2_t2);
  T* df_host_pinned_d2_v2 = (T*)getHostMem(sizeof(double) * size_T_d2_v2);

  //
  int* df_simple_s1_size = (int*)getHostMem(sizeof(int) * (6));
  int* df_simple_s1_exec = (int*)getHostMem(sizeof(int) * (9));

  int* df_simple_d1_size = (int*)getHostMem(sizeof(int) * (7 * noab));
  int* df_simple_d1_exec = (int*)getHostMem(sizeof(int) * (9 * noab));

  int* df_simple_d2_size = (int*)getHostMem(sizeof(int) * (7 * nvab));
  int* df_simple_d2_exec = (int*)getHostMem(sizeof(int) * (9 * nvab));

  //
  double* df_dev_s1_t1_all = (double*)getGpuMem(sizeof(double) * size_T_s1_t1);
  double* df_dev_s1_v2_all = (double*)getGpuMem(sizeof(double) * size_T_s1_v2);
  double* df_dev_d1_t2_all = (double*)getGpuMem(sizeof(double) * size_T_d1_t2);
  double* df_dev_d1_v2_all = (double*)getGpuMem(sizeof(double) * size_T_d1_v2);
  double* df_dev_d2_t2_all = (double*)getGpuMem(sizeof(double) * size_T_d2_t2);
  double* df_dev_d2_v2_all = (double*)getGpuMem(sizeof(double) * size_T_d2_v2);
#endif
  //
  size_t max_num_blocks = sys_data.options_map.ccsd_options.ccsdt_tilesize;
  max_num_blocks = std::ceil((max_num_blocks+4-1)/4.0);

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DCPP)
  double* df_host_energies = (double*)getHostMem(sizeof(double) * std::pow(max_num_blocks, 6) * 2);
  double* df_dev_energies = (double*)getGpuMem(sizeof(double) * std::pow(max_num_blocks, 6) * 2);
#endif

  //
  int num_task = 0;
  if(!seq_h3b)
  {
    if(rank==0) {
      std::cout << "456123 parallel 6d loop variant" << std::endl;
      std::cout << "tile142563,kernel,memcpy,data,total" << std::endl;
    }
    for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
    for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
    for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) { //
    for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
    for (size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {      
      if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
          (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
        if ((!is_restricted) ||
            (k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
            k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) {   
          if (next == taskcount) {            
            // if (has_GPU==1) {
            //   initmemmodule();
            // }
            
            double factor = 1.0;
            if (is_restricted) factor = 2.0;
            if ((t_p4b == t_p5b) && (t_p5b == t_p6b)) {
              factor /= 6.0;
            } else if ((t_p4b == t_p5b) || (t_p5b == t_p6b)) {
              factor /= 2.0;
            }

            if ((t_h1b == t_h2b) && (t_h2b == t_h3b)) {
              factor /= 6.0;
            } else if ((t_h1b == t_h2b) || (t_h2b == t_h3b)) {
              factor /= 2.0;
            }

            num_task++;
            // printf ("[%s] rank: %d >> calls the gpu code\n", __func__, rank);
            ccsd_t_fully_fused_none_df_none_task(is_restricted, noab, nvab, rank, 
                                                k_spin,
                                                k_range,
                                                k_offset,
                                                d_t1, d_t2, d_v2,
                                                k_evl_sorted,
                                                // 
                                                df_host_pinned_s1_t1, df_host_pinned_s1_v2, 
                                                df_host_pinned_d1_t2, df_host_pinned_d1_v2, 
                                                df_host_pinned_d2_t2, df_host_pinned_d2_v2,
                                                df_host_energies, 
                                                // 
                                                df_simple_s1_size, df_simple_d1_size, df_simple_d2_size, 
                                                df_simple_s1_exec, df_simple_d1_exec, df_simple_d2_exec, 
                                                // 
                                                df_dev_s1_t1_all, df_dev_s1_v2_all, 
                                                df_dev_d1_t2_all, df_dev_d1_v2_all, 
                                                df_dev_d2_t2_all, df_dev_d2_v2_all, 
                                                df_dev_energies,
                                                // 
                                                t_h1b, t_h2b, t_h3b,
                                                t_p4b, t_p5b, t_p6b,
                                                factor, taskcount, 
                                                max_d1_kernels_pertask, max_d2_kernels_pertask,
                                                //  
                                                size_T_s1_t1, size_T_s1_v2, 
                                                size_T_d1_t2, size_T_d1_v2, 
                                                size_T_d2_t2, size_T_d2_v2, 
                                                // 
                                                tmp_energy_l, 
                                                cache_s1t, cache_s1v,
                                                cache_d1t, cache_d1v,
                                                cache_d2t, cache_d2v);

            next = ac->fetch_add(0, 1);
          }
          taskcount++;
        }
      }
    }}}}}}
  } // parallel h3b loop
  else
  { //seq h3b loop
    #if 1
    if(rank==0) {
      std::cout << "14256-seq3 loop variant" << std::endl;
      std::cout << "tile142563,kernel,memcpy,data,total" << std::endl;
    }
    for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) { //    
    for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {    
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
    for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
    #endif  

    #if 0
    for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
    for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
    for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
    for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
    #endif
      if (next == taskcount) {
        // if (has_GPU==1) {
        //   initmemmodule();
        // }
        for (size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
          if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
              (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
            if ((!is_restricted) ||
                (k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
                 k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) 
            {
              double factor = 1.0;
              if (is_restricted) factor = 2.0;

              //
              if ((t_p4b == t_p5b) && (t_p5b == t_p6b)) {
                factor /= 6.0;
              } else if ((t_p4b == t_p5b) || (t_p5b == t_p6b)) {
                factor /= 2.0;
              }

              if ((t_h1b == t_h2b) && (t_h2b == t_h3b)) {
                factor /= 6.0;
              } else if ((t_h1b == t_h2b) || (t_h2b == t_h3b)) {
                factor /= 2.0;
              }

              //
              num_task++;
              ccsd_t_fully_fused_none_df_none_task(is_restricted, noab, nvab, rank, 
                                                  k_spin,
                                                  k_range,
                                                  k_offset,
                                                  d_t1, d_t2, d_v2,
                                                  k_evl_sorted,
                                                  // 
                                                  df_host_pinned_s1_t1, df_host_pinned_s1_v2, 
                                                  df_host_pinned_d1_t2, df_host_pinned_d1_v2, 
                                                  df_host_pinned_d2_t2, df_host_pinned_d2_v2, 
                                                  df_host_energies,
                                                  // 
                                                  df_simple_s1_size, df_simple_d1_size, df_simple_d2_size, 
                                                  df_simple_s1_exec, df_simple_d1_exec, df_simple_d2_exec, 
                                                  // 
                                                  df_dev_s1_t1_all, df_dev_s1_v2_all, 
                                                  df_dev_d1_t2_all, df_dev_d1_v2_all, 
                                                  df_dev_d2_t2_all, df_dev_d2_v2_all, 
                                                  df_dev_energies,
                                                  // 
                                                  t_h1b, t_h2b, t_h3b,
                                                  t_p4b, t_p5b, t_p6b,
                                                  factor, taskcount,
                                                  max_d1_kernels_pertask, max_d2_kernels_pertask,
                                                  //  
                                                  size_T_s1_t1, size_T_s1_v2, 
                                                  size_T_d1_t2, size_T_d1_v2, 
                                                  size_T_d2_t2, size_T_d2_v2, 
                                                  // 
                                                  tmp_energy_l, 
                                                  cache_s1t, cache_s1v,
                                                  cache_d1t, cache_d1v,
                                                  cache_d2t, cache_d2v);
            }
          }
        }//h3b
        // finalizememmodule();
        next = ac->fetch_add(0, 1);
      }
      taskcount++;
    }}}}}
  } //end seq h3b

  //
  energy1 = tmp_energy_l[0];
  energy2 = tmp_energy_l[1];

  //
  //
  //  free shared device mem
  //
  freeGpuMem(df_dev_s1_t1_all); freeGpuMem(df_dev_s1_v2_all);
  freeGpuMem(df_dev_d1_t2_all); freeGpuMem(df_dev_d1_v2_all);
  freeGpuMem(df_dev_d2_t2_all); freeGpuMem(df_dev_d2_v2_all);
  freeGpuMem(df_dev_energies);

  // 
  //  free shared host mem.
  // 
  freeHostMem(df_host_pinned_s1_t1);
  freeHostMem(df_host_pinned_s1_v2);
  freeHostMem(df_host_pinned_d1_t2);
  freeHostMem(df_host_pinned_d1_v2);
  freeHostMem(df_host_pinned_d2_t2);
  freeHostMem(df_host_pinned_d2_v2);
  freeHostMem(df_host_energies);

  // 
  freeHostMem(df_simple_s1_exec);
  freeHostMem(df_simple_s1_size);
  freeHostMem(df_simple_d1_exec);
  freeHostMem(df_simple_d1_size);
  freeHostMem(df_simple_d2_exec);
  freeHostMem(df_simple_d2_size);

  // 
  finalizememmodule();

  auto cc_t2 = std::chrono::high_resolution_clock::now();
  auto ccsd_t_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

  ec.pg().barrier ();
  cc_t2 = std::chrono::high_resolution_clock::now();
  auto total_t_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

  //
  next = ac->fetch_add(0, 1);
  ac->deallocate();
  delete ac;

  return std::make_tuple(energy1,energy2,ccsd_t_time,total_t_time);
}

template<typename T>
void ccsd_t_fused_driver_calculator_ops(SystemData& sys_data, ExecutionContext& ec,
                                        std::vector<int>& k_spin,
                                        const TiledIndexSpace& MO,
                                        std::vector<T>& k_evl_sorted,
                                        double hf_ccsd_energy, int iDevice,
                                        bool is_restricted,
                                        long double& total_num_ops,
                                        //
                                        bool seq_h3b=false)
{
  //
  auto rank = ec.pg().rank().value();

  Index noab=MO("occ").num_tiles();
  Index nvab=MO("virt").num_tiles();

  auto mo_tiles = MO.input_tile_sizes();
  std::vector<size_t> k_range;
  std::vector<size_t> k_offset;
  size_t sum = 0;
  for (auto x: mo_tiles){
    k_range.push_back(x);
    k_offset.push_back(sum);
    sum+=x;
  } 

  // 
  //  "list of tasks": size (t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b), factor, 
  // 
  std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, T>> list_tasks;
  if(!seq_h3b) 
  {
    for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
    for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
    for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) { //
    for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
    for (size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
      // 
      if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
          (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
        if ((!is_restricted) || 
            (k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
            k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) {
          //  
          double factor = 1.0;
          if (is_restricted) factor = 2.0;
          if ((t_p4b == t_p5b) && (t_p5b == t_p6b)) {
            factor /= 6.0;
          } else if ((t_p4b == t_p5b) || (t_p5b == t_p6b)) {
            factor /= 2.0;
          }

          if ((t_h1b == t_h2b) && (t_h2b == t_h3b)) {
            factor /= 6.0;
          } else if ((t_h1b == t_h2b) || (t_h2b == t_h3b)) {
            factor /= 2.0;
          }

          //
          list_tasks.push_back(std::make_tuple(t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, factor));
        }
      }
    }}}}}}  // nested for loops
  } // parallel h3b loop
  else
  { //seq h3b loop
    for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
    for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
    for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
    for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
      // 
      for (size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
        if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
           (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
         if ((!is_restricted) || 
             (k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
              k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) 
          {
            double factor = 1.0;
            if (is_restricted) factor = 2.0;
            // 
            if ((t_p4b == t_p5b) && (t_p5b == t_p6b)) {
              factor /= 6.0;
            } else if ((t_p4b == t_p5b) || (t_p5b == t_p6b)) {
              factor /= 2.0;
            }

            if ((t_h1b == t_h2b) && (t_h2b == t_h3b)) {
              factor /= 6.0;
            } else if ((t_h1b == t_h2b) || (t_h2b == t_h3b)) {
              factor /= 2.0;
            }
            
            // 
            list_tasks.push_back(std::make_tuple(t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, factor));
          }
        }
      }//h3b
    }}}}} // nested for loops
  } //end seq h3b 

  // 
  // 
  // 
  // printf ("[%s] rank: %d >> # of tasks: %lu\n", __func__, rank, list_tasks.size());

  // 
  //    
  // 
  total_num_ops = (long double) ccsd_t_fully_fused_performance(is_restricted,list_tasks, 
                                rank, 1, 
                                noab, nvab,
                                k_spin,k_range,k_offset,
                                k_evl_sorted);
}
#endif //CCSD_T_FUSED_HPP_