For generate_multiple the minimizations are already in the input section as: min_type
      """if run_temp: 
            sampler.run("run 1000")

        if fire_min:
            sampler.run("min_style fire")"""
          # sampler.run("""min_modify integrator eulerexplicit tmax 10.0 tmin 0.0 delaystep 5 dtgrow 1.1 dtshrink 0.5 alpha0 0.1 alphashrink 0.99 vdfmax 100000 halfstepback no initialdelay no""")
        """if line_min:
            sampler.run("min_style cg")
            sampler.run("min_modify dmax 0.05 line quadratic")"""

soft strength is in input
where are these used:     em.K_cross = cross_weight  #USED for sampler, where does sampler use em? 
        em.K_self = self_weight 

what is the 'em' model 
lammps.mliap.load_model(em)

(grs_modl) = model loaded in grs_protocol.py  -- should use grs protocol not gsqs protocol? 