# OrbitEstimator_fromUV
[2024] Article:
https://www.sciencedirect.com/science/article/abs/pii/S2213133724000702

1) To test orbit construction for initial 10 tests and EHI, change lines 115-116 in file 'UVplaneGuessor':
   
  ifRA = 0
  
  filename = 'testBlurredUV/[...].dat'

where [...] = 01, 02, ..., 10, EHI 

and compile:

  python3 UVplaneGuessor.py

plotting in 'minimize' function has 3 options:

--- plotting = 1: residuals

--- plotting = 2: 3D orbits (real + optimized)

--- plotting = 3: graphs in the article

2) To test Radioastron results:
   
--- perform function 'RA_direct_problem()' in 'radioastro_utils.py'

--- change lines 115-116 in file 'UVplaneGuessor':

   ifRA = 1
  
   filename = 'testBlurredUV/RA.dat'

and compile:

python3 UVplaneGuessor.py
