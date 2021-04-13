# Wrapper for `est_phase_sens_maps_wrapper.m`
import matlab.engine

eng = matlab.engine.start_matlab()
eng.cd('./lpr')
eng.est_phase_sens_maps_wrapper(nargout=0)

