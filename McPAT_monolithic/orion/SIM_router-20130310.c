/*-------------------------------------------------------------------------
 *                              FinCANON 
 *
 *							Copyright 2013 
 *						Princeton University
 *                         All Rights Reserved
 *
 *                         
 *  FinCANON was developed by Chun-Yi Lee, and revised by Aoxiang 
 *  Tang at Princeton University, Princeton. FinCANON was built on 
 *  top of CACTI 6.5 and ORION 2.0. 
 *
 *  If your use of this software contributes to a published paper, we
 *  request that you cite our paper: "FinCANON: A PVT-Aware 
 *  Integrated Delay and Power Modeling Framework for FinFET-based 
 *  Caches and On-Chip Networks," IEEE Transactions on VLSI Systems.
 *
 *  Permission to use, copy, and modify this software and its documentation is
 *  granted only under the following terms and conditions.  Both the
 *  above copyright notice and this permission notice must appear in all copies
 *  of the software, derivative works or modified versions, and any portions
 *  thereof, and both notices must appear in supporting documentation.
 *
 *  This software may be distributed (but not offered for sale or transferred
 *  for compensation) to third parties, provided such third parties agree to
 *  abide by the terms and conditions of this notice.
 *
 *  This software is distributed in the hope that it will be useful to the
 *  community, but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
 *
 *-----------------------------------------------------------------------*/
/*-------------------------------------------------------------------------
 *                             ORION 2.0 
 *
 *         					Copyright 2009 
 *  	Princeton University, and Regents of the University of California 
 *                         All Rights Reserved
 *
 *                         
 *  ORION 2.0 was developed by Bin Li at Princeton University and Kambiz Samadi at
 *  University of California, San Diego. ORION 2.0 was built on top of ORION 1.0. 
 *  ORION 1.0 was developed by Hangsheng Wang, Xinping Zhu and Xuning Chen at 
 *  Princeton University.
 *
 *  If your use of this software contributes to a published paper, we
 *  request that you cite our paper that appears on our website 
 *  http://www.princeton.edu/~peh/orion.html
 *
 *  Permission to use, copy, and modify this software and its documentation is
 *  granted only under the following terms and conditions.  Both the
 *  above copyright notice and this permission notice must appear in all copies
 *  of the software, derivative works or modified versions, and any portions
 *  thereof, and both notices must appear in supporting documentation.
 *
 *  This software may be distributed (but not offered for sale or transferred
 *  for compensation) to third parties, provided such third parties agree to
 *  abide by the terms and conditions of this notice.
 *
 *  This software is distributed in the hope that it will be useful to the
 *  community, but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
 *
 *-----------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "SIM_router.h"
#include "SIM_util.h"
#include "SIM_port.h"

/* global variables */
//GLOBDEF(SIM_router_power_t, router_power);
//GLOBDEF(SIM_router_info_t, router_info);
//GLOBDEF(SIM_router_area_t, router_area);

int SIM_router_init(SIM_router_info_t *info, SIM_router_power_t *router_power, SIM_router_area_t *router_area)
{
	u_int line_width;
	int share_buf, outdrv;
	InputParameter* IP = new InputParameter;
	string infile_name("");
        infile_name = "cache.cfg";
	cout<<"The input file is: "<<infile_name<<endl<<endl;

	// CACTI-FinFET Parameters ---------------------------------------------------
	IP->parse_cfg(infile_name);
	
	if(!IP->error_checking())
	{
		cout<<"IP error checking failed"<<endl;
		exit(0);
	}

	TechnologyParameter* TP = NULL;
	DynamicParameter * DP = NULL;

	TP = new TechnologyParameter(IP);
	DP = new DynamicParameter(IP, TP, /*is_tag*/0, /*pure_ram*/0, /*pure_cam*/0, IP->nspd, IP->ndwl, IP->ndbl, IP->ndcm, IP->ndsam1, IP->ndsam2, 0); // TODO: Revisit later

	// FinFET design library -----------------------------------------------------
	FinFET_Design_Library * FinFET_lib = new FinFET_Design_Library(IP, TP);

	// Process variation parameters ----------------------------------------------
	PV_parameters PVP;
	PVP.initialize(IP->Bank_PVP);

	/* PHASE 1: set configuration parameters, These are read from SIM_port.h */
	/* general parameters */
	info->n_in = PARM(in_port);
	info->n_cache_in = PARM(cache_in_port);
	info->n_mc_in = PARM(mc_in_port);
	info->n_io_in = PARM(io_in_port);
	info->n_total_in = PARM(in_port) + PARM(cache_in_port) + PARM(mc_in_port) + PARM(io_in_port);
	info->n_out = PARM(out_port);
	info->n_cache_out = PARM(cache_out_port);
	info->n_mc_out = PARM(mc_out_port);
	info->n_io_out = PARM(io_out_port);
	info->n_total_out = PARM(out_port) + PARM(cache_out_port) + PARM(mc_out_port) + PARM(io_out_port);
	info->flit_width = PARM(flit_width);

	/* virtual channel parameters */
	info->n_v_channel = MAX(PARM(v_channel), 1);
	info->n_v_class = MAX(PARM(v_class), 1); 
	info->cache_class = MAX(PARM(cache_class), 1);
	info->mc_class = MAX(PARM(mc_class), 1);
	info->io_class = MAX(PARM(io_class), 1);

	/* shared buffer implies buffer has tags */
	/* separate buffer & shared switch implies buffer has tri-state output driver*/
	if (info->n_v_class * info->n_v_channel > 1) {
		info->in_share_buf = PARM(in_share_buf);
		info->out_share_buf = PARM(out_share_buf);
		info->in_share_switch = PARM(in_share_switch);
		info->out_share_switch = PARM(out_share_switch);
	}
	else {
		info->in_share_buf = 0;
		info->out_share_buf = 0;
		info->in_share_switch = 0;
		info->out_share_switch = 0;
	}

	/* crossbar */
	info->crossbar_model = PARM(crossbar_model);
	info->degree = PARM(crsbar_degree);
	info->connect_type = PARM(connect_type);
	info->trans_type = PARM(trans_type);
	info->xb_in_seg = PARM(xb_in_seg);
	info->xb_out_seg = PARM(xb_out_seg);
	info->crossbar_in_len = PARM(crossbar_in_len);
	info->crossbar_out_len = PARM(crossbar_out_len);

	/* HACK HACK HACK */
	info->exp_xb_model = PARM(exp_xb_model);
	info->exp_in_seg = PARM(exp_in_seg);
	info->exp_out_seg = PARM(exp_out_seg);

	/* input buffer */
	// TODO: Floating point exception problem need to be fixed for SIM_array_init
	info->in_buf = PARM(in_buf);
	info->in_buffer_model = PARM(in_buffer_type);

	if(info->in_buf)
	{
		outdrv = !info->in_share_buf && info->in_share_switch;
		// TODO: SIM_array_init(&info->in_buf_info, 1, PARM(in_buf_rport), 1, PARM(in_buf_set), PARM(flit_width), outdrv, info->in_buffer_model);
	}

	if (PARM(cache_in_port))
	{
    		info->cache_in_buf = PARM(cache_in_buf);

		if (info->cache_in_buf){
			if (PARM(cache_class) > 1){
    				share_buf = info->in_share_buf;
    				outdrv = !share_buf && info->in_share_switch;
			}	
			else{
    				outdrv = share_buf = 0;
			}

    			// TODO: SIM_array_init(&info->cache_in_buf_info, 1, PARM(cache_in_buf_rport), 1, PARM(cache_in_buf_set), PARM(flit_width), outdrv, SRAM);
		}
	}

	if (PARM(mc_in_port))
	{
    		info->mc_in_buf = PARM(mc_in_buf);

		if (info->mc_in_buf){
			if (PARM(mc_class) > 1){
    				share_buf = info->in_share_buf;
    				outdrv = !share_buf && info->in_share_switch;
			}
			else{
    				outdrv = share_buf = 0;
			}

    			//TODO: SIM_array_init(&info->mc_in_buf_info, 1, PARM(mc_in_buf_rport), 1, PARM(mc_in_buf_set), PARM(flit_width), outdrv, SRAM);
		}
	}

	if (PARM(io_in_port))
	{
    		info->io_in_buf = PARM(io_in_buf);

		if (info->io_in_buf){
			if (PARM(io_class) > 1){
    				share_buf = info->in_share_buf;
    				outdrv = !share_buf && info->in_share_switch;
			}
			else{
    				outdrv = share_buf = 0;
			}

    			//TODO: SIM_array_init(&info->io_in_buf_info, 1, PARM(io_in_buf_rport), 1, PARM(io_in_buf_set), PARM(flit_width), outdrv, SRAM);
		}
	}

	/* output buffer */
	info->out_buf = PARM(out_buf);
	info->out_buffer_model = PARM(out_buffer_type);

	if (info->out_buf)
	{
		// output buffer has no tri-state buffer anyway 
		// TODO: SIM_array_init(&info->out_buf_info, 1, 1, PARM(out_buf_wport), PARM(out_buf_set), PARM(flit_width), 0, info->out_buffer_model);
	}

	/* central buffer */

	info->central_buf = PARM(central_buf);

	if (info->central_buf)
	{
		info->pipe_depth = PARM(pipe_depth);
		// central buffer is no FIFO
		// TODO: SIM_array_init(&info->central_buf_info, 0, PARM(cbuf_rport), PARM(cbuf_wport), PARM(cbuf_set), PARM(cbuf_width) * PARM(flit_width), 0, SRAM);
		// dirty hack 
		info->cbuf_ff_model = NEG_DFF;
	}

	/* switch allocator input port arbiter */

	if (info->n_v_class * info->n_v_channel > 1) 
	{
		if (info->sw_in_arb_model = PARM(sw_in_arb_model))
		{
			if (PARM(sw_in_arb_model) == QUEUE_ARBITER) 
			{
				//TODO: SIM_array_init(&info->sw_in_arb_queue_info, 1, 1, 1, info->n_v_class*info->n_v_channel, SIM_logtwo(info->n_v_class*info->n_v_channel), 0, REGISTER);
				if (info->cache_class > 1)
					// TODO: SIM_array_init(&info->cache_in_arb_queue_info, 1, 1, 1, info->cache_class, SIM_logtwo(info->cache_class), 0, REGISTER);
				if (info->mc_class > 1)
					// TODO: SIM_array_init(&info->mc_in_arb_queue_info, 1, 1, 1, info->mc_class, SIM_logtwo(info->mc_class), 0, REGISTER);
				if (info->io_class > 1)
					// TODO: SIM_array_init(&info->io_in_arb_queue_info, 1, 1, 1, info->io_class, SIM_logtwo(info->io_class), 0, REGISTER);

				info->sw_in_arb_ff_model = SIM_NO_MODEL;
			}
			else
				info->sw_in_arb_ff_model = PARM(sw_in_arb_ff_model);
		}
		else
			info->sw_in_arb_ff_model = SIM_NO_MODEL;
	}
	else 
	{
		info->sw_in_arb_model = SIM_NO_MODEL;
		info->sw_in_arb_ff_model = SIM_NO_MODEL;
	}

	/* switch allocator output port arbiter */

	if(info->n_total_in > 2)
	{
		info->sw_out_arb_model = PARM(sw_out_arb_model);

		if (info->sw_out_arb_model) 
		{
			if (info->sw_out_arb_model == QUEUE_ARBITER) 
			{
				line_width = SIM_logtwo(info->n_total_in - 1);
				//TODO: SIM_array_init(&info->sw_out_arb_queue_info, 1, 1, 1, info->n_total_in - 1, line_width, 0, REGISTER);
				info->sw_out_arb_ff_model = SIM_NO_MODEL;
			}
			else
			{
				info->sw_out_arb_ff_model = PARM(sw_out_arb_ff_model);
			}
		}
		else
		{
			info->sw_out_arb_ff_model = SIM_NO_MODEL;
		}
	}
	else
	{
		info->sw_out_arb_model = SIM_NO_MODEL;
		info->sw_out_arb_ff_model = SIM_NO_MODEL;
	}

	/* virtual channel allocator type */
	if (info->n_v_channel > 1)
	{
		info->vc_allocator_type = PARM(vc_allocator_type);
	} 
	else
		info->vc_allocator_type = SIM_NO_MODEL;

	/* virtual channel allocator input port arbiter */
	if ( info->n_v_channel > 1 && info->n_in > 1) 
	{
		if (info->vc_in_arb_model = PARM(vc_in_arb_model)) 
		{
			if (PARM(vc_in_arb_model) == QUEUE_ARBITER) 
			{ 
				// TODO: SIM_array_init(&info->vc_in_arb_queue_info, 1, 1, 1, info->n_v_channel, SIM_logtwo(info->n_v_channel), 0, REGISTER);
				info->vc_in_arb_ff_model = SIM_NO_MODEL;
			}
			else
			{
				info->vc_in_arb_ff_model = PARM(vc_in_arb_ff_model);
			}
		}
		else 
		{
			info->vc_in_arb_ff_model = SIM_NO_MODEL;
		}
	}
	else 
	{
		info->vc_in_arb_model = SIM_NO_MODEL;
		info->vc_in_arb_ff_model = SIM_NO_MODEL;
	}

	/* virtual channel allocator output port arbiter */

	if(info->n_in > 1 && info->n_v_channel > 1)
	{
		info->vc_out_arb_model = PARM(vc_out_arb_model);

		if (info->vc_out_arb_model) 
		{
			if (info->vc_out_arb_model == QUEUE_ARBITER) 
			{
				line_width = SIM_logtwo((info->n_total_in - 1)*info->n_v_channel);
				// TODO: SIM_array_init(&info->vc_out_arb_queue_info, 1, 1, 1, (info->n_total_in -1) * info->n_v_channel, line_width, 0, REGISTER);
				info->vc_out_arb_ff_model = SIM_NO_MODEL;
			}
			else
			{
				info->vc_out_arb_ff_model = PARM(vc_out_arb_ff_model);
			}
		}
		else
		{
			info->vc_out_arb_ff_model = SIM_NO_MODEL;
		}
	}
	else
	{
		info->vc_out_arb_model = SIM_NO_MODEL;
		info->vc_out_arb_ff_model = SIM_NO_MODEL;
	}

	/*virtual channel allocation vc selection model */
	info->vc_select_buf_type = PARM(vc_select_buf_type);

	if(info->vc_allocator_type == VC_SELECT && info->n_v_channel > 1 && info->n_in > 1)
	{
		info->vc_select_buf_type = PARM(vc_select_buf_type);
		// TODO: SIM_array_init(&info->vc_select_buf_info, 1, 1, 1, info->n_v_channel, SIM_logtwo(info->n_v_channel), 0, info->vc_select_buf_type);
	}
	else{
		info->vc_select_buf_type = SIM_NO_MODEL;
	}


	/* redundant fields */
	if (info->in_buf) 
	{
		if (info->in_share_buf)
			info->in_n_switch = info->in_buf_info.read_ports;
		else if (info->in_share_switch)
			info->in_n_switch = 1;
		else
			info->in_n_switch = info->n_v_class * info->n_v_channel;
	}
	else
		info->in_n_switch = 1;

	if (info->cache_in_buf) 
	{
		if (info->in_share_buf)
			info->cache_n_switch = info->cache_in_buf_info.read_ports;
		else if (info->in_share_switch)
			info->cache_n_switch = 1;
		else
			info->cache_n_switch = info->cache_class;
	}
	else
		info->cache_n_switch = 1;

	if (info->mc_in_buf) 
	{
		if (info->in_share_buf)
			info->mc_n_switch = info->mc_in_buf_info.read_ports;
		else if (info->in_share_switch)
			info->mc_n_switch = 1;
		else
			info->mc_n_switch = info->mc_class;
	}
	else
		info->mc_n_switch = 1;

	if (info->io_in_buf) 
	{
		if (info->in_share_buf)
			info->io_n_switch = info->io_in_buf_info.read_ports;
		else if (info->in_share_switch)
			info->io_n_switch = 1;
		else
			info->io_n_switch = info->io_class;
	}
	else
		info->io_n_switch = 1;

	info->n_switch_in = info->n_in * info->in_n_switch + info->n_cache_in * info->cache_n_switch +
		info->n_mc_in * info->mc_n_switch + info->n_io_in * info->io_n_switch;

	/* no buffering for local output ports */

	info->n_switch_out = info->n_cache_out + info->n_mc_out + info->n_io_out;

	if (info->out_buf) 
	{
		if (info->out_share_buf)
			info->n_switch_out += info->n_out * info->out_buf_info.write_ports;
		else if (info->out_share_switch)
			info->n_switch_out += info->n_out;
		else
			info->n_switch_out += info->n_out * info->n_v_class * info->n_v_channel;
	}
	else
		info->n_switch_out += info->n_out;

	/* clock related parameters */	

	info->pipeline_stages = PARM(pipeline_stages);
	info->H_tree_clock = PARM(H_tree_clock);
	info->router_diagonal = PARM(router_diagonal);

	/* PHASE 2: initialization */
	// Initialize the router components
	if(router_power)
	{
		router_power->IP = IP;
		router_power->TP = TP;
		router_power->DP = DP;
		router_power->FinFET_lib = FinFET_lib;
		router_power->PVP = PVP;
		SIM_router_power_init(info, router_power);
	}

	/* PHASE 3: Power estimation */
	// Calculate the power consumption of router components
	if(router_power)
	{
		
	}

	if(router_area)
	{
		SIM_router_area_init(info, router_area);
	}

	return 0;
}

ORION_Router :: ORION_Router()
{
	u_int line_width;
	int share_buf, outdrv;
	IP = new InputParameter;
	string infile_name("");
    infile_name = "cache.cfg";
	cout<<"ORION ROUTER, The input file is: "<<infile_name<<endl<<endl;

	// CACTI-FinFET Parameters ---------------------------------------------------
	IP->parse_cfg(infile_name);
	
	if(!IP->error_checking())
	{
		cout<<"IP error checking failed"<<endl;
		exit(0);
	}

	TP = new TechnologyParameter(IP);
	DP = new DynamicParameter(IP, TP, /*is_tag*/0, /*pure_ram*/0, /*pure_cam*/0, IP->nspd, IP->ndwl, IP->ndbl, IP->ndcm, IP->ndsam1, IP->ndsam2, 0); // TODO: Revisit later

	// FinFET design library -----------------------------------------------------
	FinFET_lib = new FinFET_Design_Library(IP, TP);

	// Process variation parameters ----------------------------------------------
	PVP.initialize(IP->Bank_PVP);

	/* PHASE 1: set configuration parameters, These are read from SIM_port.h */
	/* general parameters */
	router_info.n_in = PARM(in_port);
	router_info.n_cache_in = PARM(cache_in_port);
	router_info.n_mc_in = PARM(mc_in_port);
	router_info.n_io_in = PARM(io_in_port);
	router_info.n_total_in = PARM(in_port) + PARM(cache_in_port) + PARM(mc_in_port) + PARM(io_in_port);
	router_info.n_out = PARM(out_port);
	router_info.n_cache_out = PARM(cache_out_port);
	router_info.n_mc_out = PARM(mc_out_port);
	router_info.n_io_out = PARM(io_out_port);
	router_info.n_total_out = PARM(out_port) + PARM(cache_out_port) + PARM(mc_out_port) + PARM(io_out_port);
	router_info.flit_width = PARM(flit_width);

	/* virtual channel parameters */
	router_info.n_v_channel = MAX(PARM(v_channel), 1);
	router_info.n_v_class = MAX(PARM(v_class), 1); 
	router_info.cache_class = MAX(PARM(cache_class), 1);
	router_info.mc_class = MAX(PARM(mc_class), 1);
	router_info.io_class = MAX(PARM(io_class), 1);

	/* shared buffer implies buffer has tags */
	/* separate buffer & shared switch implies buffer has tri-state output driver*/
	if (router_info.n_v_class * router_info.n_v_channel > 1) 
	{
		router_info.in_share_buf = PARM(in_share_buf);
		router_info.out_share_buf = PARM(out_share_buf);
		router_info.in_share_switch = PARM(in_share_switch);
		router_info.out_share_switch = PARM(out_share_switch);
	}
	else 
	{
		router_info.in_share_buf = 0;
		router_info.out_share_buf = 0;
		router_info.in_share_switch = 0;
		router_info.out_share_switch = 0;
	}

	/* crossbar */
	router_info.crossbar_model = PARM(crossbar_model);
	router_info.degree = PARM(crsbar_degree);
	router_info.connect_type = PARM(connect_type);
	router_info.trans_type = PARM(trans_type);
	router_info.xb_in_seg = PARM(xb_in_seg);
	router_info.xb_out_seg = PARM(xb_out_seg);
	router_info.crossbar_in_len = PARM(crossbar_in_len);
	router_info.crossbar_out_len = PARM(crossbar_out_len);

	/* HACK HACK HACK */
	router_info.exp_xb_model = PARM(exp_xb_model);
	router_info.exp_in_seg = PARM(exp_in_seg);
	router_info.exp_out_seg = PARM(exp_out_seg);

	/* input buffer */
	// TODO: Floating point exception problem need to be fixed for SIM_array_init
	router_info.in_buf = PARM(in_buf);
	router_info.in_buffer_model = PARM(in_buffer_type);

	if(router_info.in_buf)
	{
		outdrv = !router_info.in_share_buf && router_info.in_share_switch;
		//it is ONLY used to initialize the values inside router_info.in_buf_info. Other computation is wasted
		SIM_array_init(&router_info.in_buf_info, 1, PARM(in_buf_rport), 1, PARM(in_buf_set), PARM(flit_width), outdrv, router_info.in_buffer_model);
	}

	if (PARM(cache_in_port))
	{
		router_info.cache_in_buf = PARM(cache_in_buf);

		if (router_info.cache_in_buf)
		{
			if (PARM(cache_class) > 1)
			{
				share_buf = router_info.in_share_buf;
				outdrv = !share_buf && router_info.in_share_switch;
			}	
			else
			{
				outdrv = share_buf = 0;
			}
			SIM_array_init(&router_info.cache_in_buf_info, 1, PARM(cache_in_buf_rport), 1, PARM(cache_in_buf_set), PARM(flit_width), outdrv, SRAM);
		}
	}

	if (PARM(mc_in_port))
	{
		router_info.mc_in_buf = PARM(mc_in_buf);

		if (router_info.mc_in_buf)
		{
			if (PARM(mc_class) > 1)
			{
				share_buf = router_info.in_share_buf;
				outdrv = !share_buf && router_info.in_share_switch;
			}
			else{
    				outdrv = share_buf = 0;
			}

    		SIM_array_init(&router_info.mc_in_buf_info, 1, PARM(mc_in_buf_rport), 1, PARM(mc_in_buf_set), PARM(flit_width), outdrv, SRAM);
		}
	}

	if (PARM(io_in_port))
	{
		router_info.io_in_buf = PARM(io_in_buf);

		if (router_info.io_in_buf)
		{
			if (PARM(io_class) > 1)
			{
				share_buf = router_info.in_share_buf;
				outdrv = !share_buf && router_info.in_share_switch;
			}
			else
			{
				outdrv = share_buf = 0;
			}

    		SIM_array_init(&router_info.io_in_buf_info, 1, PARM(io_in_buf_rport), 1, PARM(io_in_buf_set), PARM(flit_width), outdrv, SRAM);
		}
	}

	/* output buffer */
	router_info.out_buf = PARM(out_buf);
	router_info.out_buffer_model = PARM(out_buffer_type);

	if (router_info.out_buf)
	{
		// output buffer has no tri-state buffer anyway 
		SIM_array_init(&router_info.out_buf_info, 1, 1, PARM(out_buf_wport), PARM(out_buf_set), PARM(flit_width), 0, router_info.out_buffer_model);
	}

	/* central buffer */

	router_info.central_buf = PARM(central_buf);

	if (router_info.central_buf)
	{
		router_info.pipe_depth = PARM(pipe_depth);
		// central buffer is no FIFO
		SIM_array_init(&router_info.central_buf_info, 0, PARM(cbuf_rport), PARM(cbuf_wport), PARM(cbuf_set), PARM(cbuf_width) * PARM(flit_width), 0, SRAM);
		// dirty hack 
		router_info.cbuf_ff_model = NEG_DFF;
	}

	/* switch allocator input port arbiter */

	if (router_info.n_v_class * router_info.n_v_channel > 1) 
	{
		if (router_info.sw_in_arb_model = PARM(sw_in_arb_model))
		{
			if (PARM(sw_in_arb_model) == QUEUE_ARBITER) 
			{
				SIM_array_init(&router_info.sw_in_arb_queue_info, 1, 1, 1, router_info.n_v_class*router_info.n_v_channel, SIM_logtwo(router_info.n_v_class*router_info.n_v_channel), 0, REGISTER);
				if (router_info.cache_class > 1)
					SIM_array_init(&router_info.cache_in_arb_queue_info, 1, 1, 1, router_info.cache_class, SIM_logtwo(router_info.cache_class), 0, REGISTER);
				if (router_info.mc_class > 1)
					SIM_array_init(&router_info.mc_in_arb_queue_info, 1, 1, 1, router_info.mc_class, SIM_logtwo(router_info.mc_class), 0, REGISTER);
				if (router_info.io_class > 1)
					SIM_array_init(&router_info.io_in_arb_queue_info, 1, 1, 1, router_info.io_class, SIM_logtwo(router_info.io_class), 0, REGISTER);

				router_info.sw_in_arb_ff_model = SIM_NO_MODEL;
			}
			else
				router_info.sw_in_arb_ff_model = PARM(sw_in_arb_ff_model);
		}
		else
			router_info.sw_in_arb_ff_model = SIM_NO_MODEL;
	}
	else 
	{
		router_info.sw_in_arb_model = SIM_NO_MODEL;
		router_info.sw_in_arb_ff_model = SIM_NO_MODEL;
	}

	/* switch allocator output port arbiter */

	if(router_info.n_total_in > 2)
	{
		router_info.sw_out_arb_model = PARM(sw_out_arb_model);

		if (router_info.sw_out_arb_model) 
		{
			if (router_info.sw_out_arb_model == QUEUE_ARBITER) 
			{
				line_width = SIM_logtwo(router_info.n_total_in - 1);
				SIM_array_init(&router_info.sw_out_arb_queue_info, 1, 1, 1, router_info.n_total_in - 1, line_width, 0, REGISTER);
				router_info.sw_out_arb_ff_model = SIM_NO_MODEL;
			}
			else
			{
				router_info.sw_out_arb_ff_model = PARM(sw_out_arb_ff_model);
			}
		}
		else
		{
			router_info.sw_out_arb_ff_model = SIM_NO_MODEL;
		}
	}
	else
	{
		router_info.sw_out_arb_model = SIM_NO_MODEL;
		router_info.sw_out_arb_ff_model = SIM_NO_MODEL;
	}

	/* virtual channel allocator type */
	if (router_info.n_v_channel > 1)
	{
		router_info.vc_allocator_type = PARM(vc_allocator_type);
	} 
	else
		router_info.vc_allocator_type = SIM_NO_MODEL;

	/* virtual channel allocator input port arbiter */
	if ( router_info.n_v_channel > 1 && router_info.n_in > 1) 
	{
		if (router_info.vc_in_arb_model = PARM(vc_in_arb_model)) 
		{
			if (PARM(vc_in_arb_model) == QUEUE_ARBITER) 
			{ 
				SIM_array_init(&router_info.vc_in_arb_queue_info, 1, 1, 1, router_info.n_v_channel, SIM_logtwo(router_info.n_v_channel), 0, REGISTER);
				router_info.vc_in_arb_ff_model = SIM_NO_MODEL;
			}
			else
			{
				router_info.vc_in_arb_ff_model = PARM(vc_in_arb_ff_model);
			}
		}
		else 
		{
			router_info.vc_in_arb_ff_model = SIM_NO_MODEL;
		}
	}
	else 
	{
		router_info.vc_in_arb_model = SIM_NO_MODEL;
		router_info.vc_in_arb_ff_model = SIM_NO_MODEL;
	}

	/* virtual channel allocator output port arbiter */

	if(router_info.n_in > 1 && router_info.n_v_channel > 1)
	{
		router_info.vc_out_arb_model = PARM(vc_out_arb_model);

		if (router_info.vc_out_arb_model) 
		{
			if (router_info.vc_out_arb_model == QUEUE_ARBITER) 
			{
				line_width = SIM_logtwo((router_info.n_total_in - 1)*router_info.n_v_channel);
				SIM_array_init(&router_info.vc_out_arb_queue_info, 1, 1, 1, (router_info.n_total_in -1) * router_info.n_v_channel, line_width, 0, REGISTER);
				router_info.vc_out_arb_ff_model = SIM_NO_MODEL;
			}
			else
			{
				router_info.vc_out_arb_ff_model = PARM(vc_out_arb_ff_model);
			}
		}
		else
		{
			router_info.vc_out_arb_ff_model = SIM_NO_MODEL;
		}
	}
	else
	{
		router_info.vc_out_arb_model = SIM_NO_MODEL;
		router_info.vc_out_arb_ff_model = SIM_NO_MODEL;
	}

	/*virtual channel allocation vc selection model */
	router_info.vc_select_buf_type = PARM(vc_select_buf_type);

	if(router_info.vc_allocator_type == VC_SELECT && router_info.n_v_channel > 1 && router_info.n_in > 1)
	{
		router_info.vc_select_buf_type = PARM(vc_select_buf_type);
		SIM_array_init(&router_info.vc_select_buf_info, 1, 1, 1, router_info.n_v_channel, SIM_logtwo(router_info.n_v_channel), 0, router_info.vc_select_buf_type);
	}
	else{
		router_info.vc_select_buf_type = SIM_NO_MODEL;
	}


	/* redundant fields */
	if (router_info.in_buf) 
	{
		if (router_info.in_share_buf)
			router_info.in_n_switch = router_info.in_buf_info.read_ports;
		else if (router_info.in_share_switch)
			router_info.in_n_switch = 1;
		else
			router_info.in_n_switch = router_info.n_v_class * router_info.n_v_channel;
	}
	else
		router_info.in_n_switch = 1;

	if (router_info.cache_in_buf) 
	{
		if (router_info.in_share_buf)
			router_info.cache_n_switch = router_info.cache_in_buf_info.read_ports;
		else if (router_info.in_share_switch)
			router_info.cache_n_switch = 1;
		else
			router_info.cache_n_switch = router_info.cache_class;
	}
	else
		router_info.cache_n_switch = 1;

	if (router_info.mc_in_buf) 
	{
		if (router_info.in_share_buf)
			router_info.mc_n_switch = router_info.mc_in_buf_info.read_ports;
		else if (router_info.in_share_switch)
			router_info.mc_n_switch = 1;
		else
			router_info.mc_n_switch = router_info.mc_class;
	}
	else
		router_info.mc_n_switch = 1;

	if (router_info.io_in_buf) 
	{
		if (router_info.in_share_buf)
			router_info.io_n_switch = router_info.io_in_buf_info.read_ports;
		else if (router_info.in_share_switch)
			router_info.io_n_switch = 1;
		else
			router_info.io_n_switch = router_info.io_class;
	}
	else
		router_info.io_n_switch = 1;

	router_info.n_switch_in = router_info.n_in * router_info.in_n_switch + router_info.n_cache_in * router_info.cache_n_switch +
		router_info.n_mc_in * router_info.mc_n_switch + router_info.n_io_in * router_info.io_n_switch;

	/* no buffering for local output ports */

	router_info.n_switch_out = router_info.n_cache_out + router_info.n_mc_out + router_info.n_io_out;

	if (router_info.out_buf) 
	{
		if (router_info.out_share_buf)
			router_info.n_switch_out += router_info.n_out * router_info.out_buf_info.write_ports;
		else if (router_info.out_share_switch)
			router_info.n_switch_out += router_info.n_out;
		else
			router_info.n_switch_out += router_info.n_out * router_info.n_v_class * router_info.n_v_channel;
	}
	else
		router_info.n_switch_out += router_info.n_out;

	/* clock related parameters */	

	router_info.pipeline_stages = PARM(pipeline_stages);
	router_info.H_tree_clock = PARM(H_tree_clock);
	router_info.router_diagonal = PARM(router_diagonal);

	/* PHASE 2: initialization */
	// Initialize the router components
	router_power.IP = IP;
	router_power.TP = TP;
	router_power.DP = DP;
	router_power.FinFET_lib = FinFET_lib;
	router_power.PVP = PVP;
	//SIM_router_power_init(&router_info, &router_power);
	 router_power_init();

	//SIM_router_area_init(&router_info, &router_area);
	 router_area_init();
	 area.set_area(get_router_area());
	 cout << " area: "<<area.get_area()<<endl;
}

void ORION_Router :: router_power_init()
{
	router_power.Buffers_Leak.reset();
	router_power.Arbiters_Leak.reset();
	double num_macros = 0; 
    double LK = 0, Mu = 0, Var = 0;

	double cbuf_width, req_len = 0;

	router_power.I_static = 0;
	router_power.I_buf_static = 0;
	router_power.I_crossbar_static = 0;
	router_power.I_vc_arbiter_static = 0;
	router_power.I_sw_arbiter_static = 0;
	
	Token  Empty_Token;
    list<Token> TA, TD;
    Empty_Token.PVP = router_power.PVP;          // TODO: put this into the constructor of the Token class
    TA.push_back(Empty_Token);
    TD.push_back(Empty_Token);

	router_power.m_uca = new UCA(router_power.DP, router_power.IP, router_power.TP, router_power.FinFET_lib, router_power.PVP);

	cout<<"========================================================================================="<<endl<<endl;

	// initialize crossbar ---------------------------------------------------------------------------
	cout<<"[CROSSBAR]"<<endl<<endl;
	if (router_info.crossbar_model) {
		router_power.m_crossbar.init(router_power.IP, router_power.TP, router_power.FinFET_lib, router_power.PVP, router_info.crossbar_model, router_info.n_switch_in, router_info.n_switch_out, router_info.xb_in_seg, router_info.xb_out_seg, router_info.flit_width, router_info.degree, router_info.connect_type, router_info.trans_type, router_info.crossbar_in_len, router_info.crossbar_out_len, &req_len);
		
		LK = router_power.m_crossbar.XB_Leak.Mu_Sum;
		Mu = router_power.m_crossbar.XB_Leak.Mu_Z;
		Var = router_power.m_crossbar.XB_Leak.Var_Z;
		router_power.Router_Leak.insert_item(1, Mu, Var);
		// static power
		//router_power.I_crossbar_static = router_power.crossbar.I_static;
		//router_power.I_static += router_power.I_crossbar_static;
	}

	cout<<"========================================================================================="<<endl<<endl;

	cout<<"[INPUT BUFFER]"<<endl;
	// HACK HACK HACK
	if (router_info.exp_xb_model)
	{
		router_power.m_exp_xb.init(router_power.IP, router_power.TP, router_power.FinFET_lib, router_power.PVP, router_info.exp_xb_model, 2 * router_info.n_switch_in - 1, 2 * router_info.n_switch_out - 1, router_info.exp_in_seg, router_info.exp_out_seg, router_info.flit_width, router_info.degree, router_info.connect_type, router_info.trans_type, 0, 0, NULL);
		
		LK = router_power.m_exp_xb.XB_Leak.Mu_Sum;
		Mu = router_power.m_exp_xb.XB_Leak.Mu_Z;
		Var = router_power.m_exp_xb.XB_Leak.Var_Z;
		router_power.Router_Leak.insert_item(1, Mu, Var);
	}

	// TODO initialize input buffer ------------------------------------------------------------------
	// initialize various buffers
	if (router_info.in_buf) {
		router_power.m_uca->compute_delays(&TA, &TD);
		router_power.m_uca->compute_power_energy();
		cout<<"Input buffer dynamic power per access = "<<router_power.m_uca->power.readOp.dynamic<<endl;
		num_macros = router_info.n_in * router_info.n_v_class;
		LK = router_power.m_uca->UCA_Leak.Mu_Sum;
		Mu = router_power.m_uca->UCA_Leak.Mu_Z;
		Var = router_power.m_uca->UCA_Leak.Var_Z;

		router_power.Buffers_Leak.insert_item(num_macros, Mu, Var);
		router_power.Router_Leak.insert_item(num_macros, Mu, Var);

		
		router_power.Buffers_Leak.calculate_macro();
		cout<<"Buffers_Leak.Mu_Sum = "<<router_power.Buffers_Leak.Mu_Sum<<" : Buffers_Leak.Var_Sum = "<<sqrt(router_power.Buffers_Leak.Var_Sum)<<endl<<endl;
		//cout <<"Buffer area" << router_power.m_uca->area.get_area() << endl;
		router_area.buffer = router_power.m_uca->area.get_area();
		//SIM_array_power_init(&router_info.in_buf_info, &router_power.in_buf);
		// static power
		//router_power.I_buf_static = router_power.in_buf.I_static * router_info.n_in * router_info.n_v_class * (router_info.in_share_buf ? 1 : router_info.n_v_channel);
	}

	cout<<"========================================================================================="<<endl<<endl;


	//router_power.I_static += router_power.I_buf_static;

	cout<<"[ARBITER]"<<endl<<endl;
	// initialize switch allocator arbiter -----------------------------------------------------------
	cout<<"1st ARB"<<endl;
	if (router_info.sw_in_arb_model) {
		router_power.m_sw_in_arb.init(router_power.IP, router_power.TP, router_power.FinFET_lib, router_power.PVP, router_info.sw_in_arb_model, router_info.sw_in_arb_ff_model, router_info.n_v_channel*router_info.n_v_class, 0, &router_info.sw_in_arb_queue_info, (router_info.in_n_switch * router_info.n_in) );
		
		/* TODO: Temporarily not used
		if (router_info.n_cache_in)
		{
			router_power.m_cache_in_arb.init(router_power.IP, router_power.TP, router_power.FinFET_lib, router_power.PVP, router_info.sw_in_arb_model, router_info.sw_in_arb_ff_model, router_info.cache_class, 0, &router_info.cache_in_arb_queue_info);
		}

		if (router_info.n_mc_in)
		{
			router_power.m_mc_in_arb.init(router_power.IP, router_power.TP, router_power.FinFET_lib, router_power.PVP, router_info.sw_in_arb_model, router_info.sw_in_arb_ff_model, router_info.mc_class, 0, &router_info.mc_in_arb_queue_info);
		}

		if (router_info.n_io_in)
		{
			router_power.m_io_in_arb.init(router_power.IP, router_power.TP, router_power.FinFET_lib, router_power.PVP, router_info.sw_in_arb_model, router_info.sw_in_arb_ff_model, router_info.io_class, 0, &router_info.io_in_arb_queue_info);
		}
		*/

		num_macros = 1;
		LK = router_power.m_sw_in_arb.ARB_Leak.Mu_Sum;
		Mu = router_power.m_sw_in_arb.ARB_Leak.Mu_Z;
		Var = router_power.m_sw_in_arb.ARB_Leak.Var_Z;

		router_power.Arbiters_Leak.insert_item(num_macros, Mu, Var);
		router_power.Router_Leak.insert_item(num_macros, Mu, Var);
		//router_power.I_sw_arbiter_static = router_power.sw_in_arb.I_static * router_info.in_n_switch * router_info.n_in;
	}

	// WHS: must after switch initialization
	cout<<"2nd ARB"<<endl;
	if (router_info.sw_out_arb_model) {
		router_power.m_sw_out_arb.init(router_power.IP, router_power.TP, router_power.FinFET_lib, router_power.PVP, router_info.sw_out_arb_model, router_info.sw_out_arb_ff_model, router_info.n_total_in - 1, req_len, &router_info.sw_out_arb_queue_info, (router_info.n_switch_out));
		//cout<<"router_info.n_total_in = "<<router_info.n_total_in<<" : router_info.n_switch_out = "<<router_info.n_switch_out<<endl;
		router_power.I_sw_arbiter_static += router_power.sw_out_arb.I_static * router_info.n_switch_out;

		num_macros = 1;
		LK = router_power.m_sw_out_arb.ARB_Leak.Mu_Sum;
		Mu = router_power.m_sw_out_arb.ARB_Leak.Mu_Z;
		Var = router_power.m_sw_out_arb.ARB_Leak.Var_Z;

		router_power.Arbiters_Leak.insert_item(num_macros, Mu, Var);
		router_power.Router_Leak.insert_item(num_macros, Mu, Var);
	}
	/*static energy*/ 
	//router_power.I_static += router_power.I_sw_arbiter_static;


	// initialize virtual channel allocator arbiter --------------------------------------------------
	if(router_info.vc_allocator_type == ONE_STAGE_ARB && router_info.n_v_channel > 1 && router_info.n_in > 1)
	{	
		/* TODO: Temporarily not used			
		if (router_info.vc_out_arb_model)
		{
			router_power.m_vc_out_arb.init(router_power.IP, router_power.TP, router_power.FinFET_lib, router_power.PVP, router_info.vc_out_arb_model, router_info.vc_out_arb_ff_model, (router_info.n_in - 1) * router_info.n_v_channel, 0, &router_info.vc_out_arb_queue_info);

			router_power.I_vc_arbiter_static = router_power.vc_out_arb.I_static * router_info.n_out * router_info.n_v_channel * router_info.n_v_class;
		}
		else {
			fprintf (stderr, "error in setting vc allocator parameters\n");
		}
		*/
	}
	else if(router_info.vc_allocator_type == TWO_STAGE_ARB && router_info.n_v_channel > 1 && router_info.n_in > 1)
	{	// Temporarily use two stage allocator in this work!
		if (router_info.vc_in_arb_model && router_info.vc_out_arb_model)
		{	
			cout<<"3th ARB"<<endl;
			// first stage
			router_power.m_vc_in_arb.init(router_power.IP, router_power.TP, router_power.FinFET_lib, router_power.PVP, router_info.vc_in_arb_model, router_info.vc_in_arb_ff_model, router_info.n_v_channel, 0, &router_info.vc_in_arb_queue_info, (router_info.n_in * router_info.n_v_channel * router_info.n_v_class));
			//cout<<"router_info.n_in = "<<router_info.n_in<<" : router_info.n_v_channel = "<<router_info.n_v_channel<<" : router_info.n_v_class = "<<router_info.n_v_class<<endl;

			num_macros = 1;
			LK = router_power.m_vc_in_arb.ARB_Leak.Mu_Sum;
			Mu = router_power.m_vc_in_arb.ARB_Leak.Mu_Z;
			Var = router_power.m_vc_in_arb.ARB_Leak.Var_Z;

			router_power.Arbiters_Leak.insert_item(num_macros, Mu, Var);
			router_power.Router_Leak.insert_item(num_macros, Mu, Var);

			//router_power.I_vc_arbiter_static = router_power.vc_in_arb.I_static * router_info.n_in * router_info.n_v_channel * router_info.n_v_class;

			cout<<"4th ARB"<<endl;
			//second stage
			router_power.m_vc_out_arb.init(router_power.IP, router_power.TP, router_power.FinFET_lib, router_power.PVP, router_info.vc_out_arb_model, router_info.vc_out_arb_ff_model, (router_info.n_in - 1) * router_info.n_v_channel, 0, &router_info.vc_out_arb_queue_info, router_info.n_out * router_info.n_v_channel * router_info.n_v_class);

			num_macros = 1;
			LK = router_power.m_vc_out_arb.ARB_Leak.Mu_Sum;
			Mu = router_power.m_vc_out_arb.ARB_Leak.Mu_Z;
			Var = router_power.m_vc_out_arb.ARB_Leak.Var_Z;
			router_power.Arbiters_Leak.insert_item(num_macros, Mu, Var);
			router_power.Router_Leak.insert_item(num_macros, Mu, Var);

			//router_power.I_vc_arbiter_static += router_power.vc_out_arb.I_static * router_info.n_out * router_info.n_v_channel * router_info.n_v_class;
		}	
		else 
		{
			fprintf (stderr, "error in setting vc allocator parameters\n");
		}
	}
	else if(router_info.vc_allocator_type == VC_SELECT && router_info.n_v_channel > 1) 
	{	/* TODO: Temporarily not used
		// TODO: verify later
		SIM_array_power_init(&router_info.vc_select_buf_info, &router_power.vc_select_buf);
		// static power
		router_power.I_vc_arbiter_static = router_power.vc_select_buf.I_static * router_info.n_out * router_info.n_v_class;
		*/
	}
	else
	{	// Some error handler
		exit(0);
	}

	router_power.Arbiters_Leak.calculate_macro();
	cout<<"Arbiters_Leak.Mu_Sum = "<<router_power.Arbiters_Leak.Mu_Sum<<" : Arbiters_Leak.Var_Sum = "<<sqrt(router_power.Arbiters_Leak.Var_Sum)<<endl<<endl;

	cout<<"========================================================================================="<<endl<<endl;
	cout<<"[CLOCK]"<<endl<<endl;
	// Clock
	router_power.m_CLK.init(router_power.IP, router_power.TP, router_power.FinFET_lib, router_power.PVP, router_info.router_diagonal, router_info.in_buf, router_info.in_share_switch, router_info.crossbar_model, router_info.out_share_switch, router_info.out_buf, router_info.flit_width, router_info.n_total_in, router_info.n_total_out, router_info.n_v_channel, router_info.n_v_class);

	num_macros = 1;
	LK = router_power.m_CLK.CLK_Leak.Mu_Sum;
	Mu = router_power.m_CLK.CLK_Leak.Mu_Z;
	Var = router_power.m_CLK.CLK_Leak.Var_Z;
	router_power.Router_Leak.insert_item(num_macros, Mu, Var);

	router_power.Router_Leak.calculate_macro();
	cout<<"[Total: Router]"<<endl<<endl;
	cout<<"Router_Leak.Mu_Sum = "<<router_power.Router_Leak.Mu_Sum<<" : Router_Leak.Var_Sum = "<<sqrt(router_power.Router_Leak.Var_Sum)<<endl<<endl;

	power.readOp.leakage = router_power.Router_Leak.Mu_Sum;
	power.readOp.leakage_sigma = sqrt(router_power.Router_Leak.Var_Sum);
	power.readOp.lognormal_leakage = router_power.Router_Leak.Mu_Z;
	power.readOp.lognormal_leakage_sigma = sqrt(router_power.Router_Leak.Var_Z);
	//router_power.I_static += router_power.I_vc_arbiter_static;
}

void ORION_Router :: router_area_init()
{
	double bitline_len, wordline_len, xb_in_len, xb_out_len;
	double depth, nMUX, boxArea;
	int req_width;
	//buffer area is computed in router_power_init, can't be set to 0 here
	//router_area.buffer = 0;
	//cout << "hello: " << router_area.buffer << endl;
	router_area.crossbar = 0;
	router_area.vc_allocator = 0;
	router_area.sw_allocator = 0;


	/* buffer area */
	/* input buffer area */

		
	if (router_info.in_buf) 
	{
		switch (router_info.in_buffer_model) {
			case SRAM:
				//this part is computed in router_power_init();
				/*
				bitline_len = router_info.in_buf_info.n_set * (RegCellHeight + 2 * WordlineSpacing);
				wordline_len = router_info.flit_width * (RegCellWidth + 2 * (router_info.in_buf_info.read_ports 
							+ router_info.in_buf_info.write_ports) * BitlineSpacing);
				*/

				/* input buffer area */
				/*
				router_area.buffer = router_info.n_in * router_info.n_v_class * (bitline_len * wordline_len) *
										(router_info.in_share_buf ? 1 : router_info.n_v_channel );
				*/
				cout << "hello: " << router_area.buffer << endl;
				break;

			case REGISTER:
				/*
				router_area.buffer = AreaDFF * router_info.n_in * router_info.n_v_class * router_info.flit_width *
					router_info.in_buf_info.n_set * (router_info.in_share_buf ? 1 : router_info.n_v_channel );
				*/
				router_area.buffer = FinFET_lib->GDFF[0].compute_area(1) * router_info.n_in * router_info.n_v_class * router_info.flit_width *
					router_info.in_buf_info.n_set * (router_info.in_share_buf ? 1 : router_info.n_v_channel );
				break;

			default: printf ("error\n");  /* some error handler */
		}
	}

	/* output buffer area */
	if (router_info.out_buf) 
	{
		switch (router_info.out_buffer_model) {
			case SRAM:
				/*
				bitline_len = router_info.out_buf_info.n_set * (RegCellHeight + 2 * WordlineSpacing);
				wordline_len = router_info.flit_width * (RegCellWidth + 2 * (router_info.out_buf_info.read_ports 
							+ router_info.out_buf_info.write_ports) * BitlineSpacing);
				*/

				/* output buffer area */
				/*
				router_area.buffer += router_info.n_out * router_info.n_v_class * (bitline_len * wordline_len) *
									(router_info.out_share_buf ? 1 : router_info.n_v_channel );
				*/
				break;

			case REGISTER:
				/*
				router_area.buffer += AreaDFF * router_info.n_out * router_info.n_v_class * router_info.flit_width * 
					router_info.out_buf_info.n_set * (router_info.out_share_buf ? 1 : router_info.n_v_channel ) ; 
				*/
				router_area.buffer +=FinFET_lib->GDFF[0].compute_area(1) * router_info.n_out * router_info.n_v_class * router_info.flit_width * 
					router_info.out_buf_info.n_set * (router_info.out_share_buf ? 1 : router_info.n_v_channel ) ; 
				break;

			default: printf ("error\n");  /* some error handler */
		}
	}

	/* crossbar area */
	if (router_info.crossbar_model && router_info.crossbar_model < CROSSBAR_MAX_MODEL) 
	{
		switch (router_info.crossbar_model) {
			case MATRIX_CROSSBAR:
				/*
				xb_in_len = router_info.n_switch_in * router_info.flit_width * CrsbarCellWidth;  
				xb_out_len = router_info.n_switch_out * router_info.flit_width * CrsbarCellHeight;
				*/
				xb_in_len = router_info.n_switch_in * router_info.flit_width * (FinFET_lib->GINV[0].Width + FinFET_lib->GNOR2[0].Width + FinFET_lib->GNAND2[0].Width);  
				xb_out_len = router_info.n_switch_out * router_info.flit_width * (FinFET_lib->GINV[0].Height);  
				router_area.crossbar = xb_in_len * xb_out_len;
				break;
			//not done yet
			case MULTREE_CROSSBAR:
				/*
				if(router_info.degree == 2) {
					depth = ceil((log(router_info.n_switch_in) / log(2)));  
					nMUX = pow(2,depth) - 1;
					boxArea = 1.5 *nMUX * AreaMUX2;
					router_area.crossbar = router_info.n_switch_in * router_info.flit_width *boxArea * router_info.n_switch_out; 
				}
				else if( router_info.degree == 3 ) {
					depth = ceil((log(router_info.n_switch_in) / log(3))); 
					nMUX = ((pow(3,depth) - 1) / 2);
					boxArea = 1.5 * nMUX * AreaMUX3;
					router_area.crossbar = router_info.n_switch_in * router_info.flit_width *boxArea * router_info.n_switch_out; 
				}
				else if( router_info.degree == 4 ) {
					depth = ceil((log(router_info.n_switch_in) / log(4)));
					nMUX = ((pow(4,depth) - 1) / 3);
					boxArea = 1.5 * nMUX * AreaMUX4;
					router_area.crossbar = router_info.n_switch_in * router_info.flit_width * boxArea * router_info.n_switch_out; 
				}
				*/
				break;

			default: printf ("error\n");  /* some error handler */

		}
	}

	if (router_info.exp_xb_model)
	{ //only support for MATRIX_CROSSBAR type
		/*
		xb_in_len = (2 *router_info.n_switch_in - 1) * router_info.flit_width * CrsbarCellWidth; 
		xb_out_len = (2 * router_info.n_switch_out - 1) * router_info.flit_width * CrsbarCellHeight; 
		*/
		//width of a tristate buffer: a nand2, a nor2 and a inverter
		xb_in_len = (2 *router_info.n_switch_in - 1) * router_info.flit_width * (FinFET_lib->GINV[0].Width + FinFET_lib->GNOR2[0].Width + FinFET_lib->GNAND2[0].Width);  
		xb_out_len = (2 * router_info.n_switch_out - 1) * router_info.flit_width *  (FinFET_lib->GINV[0].Height);  
		
		router_area.crossbar = xb_in_len * xb_out_len;
	}

	/* switch allocator area */
	if (router_info.sw_in_arb_model) 
	{
		req_width = router_info.n_v_channel * router_info.n_v_class;

		switch (router_info.sw_in_arb_model) {	
			case MATRIX_ARBITER:  //assumes 30% spacing for each arbiter
				/*
				router_area.sw_allocator += ((AreaNOR * 2 * (req_width - 1) * req_width) + (AreaINV * req_width) 
						+ (AreaDFF * (req_width * (req_width - 1)/2))) * 1.3 * router_info.in_n_switch * router_info.n_in;
				*/
				router_area.sw_allocator += ((FinFET_lib->GNOR2[0].compute_area(1) * 2 * (req_width - 1) * req_width) + (FinFET_lib->GINV[0].compute_area(1)  * req_width) 
						+ (FinFET_lib->GDFF[0].compute_area(1)  * (req_width * (req_width - 1)/2))) * 1.3 * router_info.in_n_switch * router_info.n_in;
				break;

			case RR_ARBITER: //assumes 30% spacing for each arbiter
				/*
				router_area.sw_allocator += ((6 *req_width * AreaNOR) + (2 * req_width * AreaINV) 
											+ (req_width * AreaDFF)) * 1.3 * router_info.in_n_switch * router_info.n_in;
				*/
				router_area.sw_allocator += ((6 *req_width * FinFET_lib->GNOR2[0].compute_area(1)) + (2 * req_width * FinFET_lib->GINV[0].compute_area(1)) 
											+ (req_width * FinFET_lib->GDFF[0].compute_area(1))) * 1.3 * router_info.in_n_switch * router_info.n_in;
				break;

			case QUEUE_ARBITER: 
				/*
				router_area.sw_allocator += AreaDFF * router_info.sw_in_arb_queue_info.n_set * router_info.sw_in_arb_queue_info.data_width
					* router_info.in_n_switch * router_info.n_in;
				*/
				router_area.sw_allocator += FinFET_lib->GDFF[0].compute_area(1) * router_info.sw_in_arb_queue_info.n_set * router_info.sw_in_arb_queue_info.data_width
					* router_info.in_n_switch * router_info.n_in;

				break;

			default: printf ("error\n");  /* some error handler */	
		}
	}

	if (router_info.sw_out_arb_model) 
	{
		req_width = router_info.n_total_in - 1;

		switch (router_info.sw_out_arb_model) {
			case MATRIX_ARBITER: //assumes 30% spacing for each arbiter
				/*
				router_area.sw_allocator += ((AreaNOR * 2 * (req_width - 1) * req_width) + (AreaINV * req_width)
						+ (AreaDFF * (req_width * (req_width - 1)/2))) * 1.3 * router_info.n_switch_out;
				*/
				router_area.sw_allocator += ((FinFET_lib->GNOR2[0].compute_area(1) * 2 * (req_width - 1) * req_width) + (FinFET_lib->GINV[0].compute_area(1) * req_width)
						+ (FinFET_lib->GDFF[0].compute_area(1) * (req_width * (req_width - 1)/2))) * 1.3 * router_info.n_switch_out;
				break;

			case RR_ARBITER: //assumes 30% spacing for each arbiter
				/*
				router_area.sw_allocator += ((6 *req_width * AreaNOR) + (2 * req_width * AreaINV) + (req_width * AreaDFF)) * 1.3 * router_info.n_switch_out;
				*/
				router_area.sw_allocator += ((6 *req_width * FinFET_lib->GNOR2[0].compute_area(1)) + (2 * req_width * FinFET_lib->GINV[0].compute_area(1)) + (req_width * FinFET_lib->GDFF[0].compute_area(1))) * 1.3 * router_info.n_switch_out;
				break;

			case QUEUE_ARBITER:
				/*
				router_area.sw_allocator += AreaDFF * router_info.sw_out_arb_queue_info.data_width
					* router_info.sw_out_arb_queue_info.n_set * router_info.n_switch_out;
				*/
				router_area.sw_allocator += FinFET_lib->GDFF[0].compute_area(1) * router_info.sw_out_arb_queue_info.data_width
					* router_info.sw_out_arb_queue_info.n_set * router_info.n_switch_out;
				break;

			default: printf ("error\n");  /* some error handler */  


		}
	}


	/* virtual channel allocator area */
	if(router_info.vc_allocator_type == ONE_STAGE_ARB && router_info.n_v_channel > 1 && router_info.n_in > 1)
	{
		if (router_info.vc_out_arb_model){
			req_width = (router_info.n_in - 1) * router_info.n_v_channel;
			switch (router_info.vc_out_arb_model){
				case MATRIX_ARBITER: //assumes 30% spacing for each arbiter
					/*
					router_area.vc_allocator = ((AreaNOR * 2 * (req_width - 1) * req_width) + (AreaINV * req_width)
							+ (AreaDFF * (req_width * (req_width - 1)/2))) * 1.3 * router_info.n_out * router_info.n_v_channel * router_info.n_v_class;
					*/
					router_area.vc_allocator = ((FinFET_lib->GNOR2[0].compute_area(1) * 2 * (req_width - 1) * req_width) + (FinFET_lib->GINV[0].compute_area(1) * req_width)
							+ (FinFET_lib->GDFF[0].compute_area(1) * (req_width * (req_width - 1)/2))) * 1.3 * router_info.n_out * router_info.n_v_channel * router_info.n_v_class;
					break;

				case RR_ARBITER: //assumes 30% spacing for each arbiter
					/*
					router_area.vc_allocator = ((6 *req_width * AreaNOR) + (2 * req_width * AreaINV) + (req_width * AreaDFF)) * 1.3  
												* router_info.n_out * router_info.n_v_channel * router_info.n_v_class;
					*/
					router_area.vc_allocator = ((6 *req_width * FinFET_lib->GNOR2[0].compute_area(1)) + (2 * req_width * FinFET_lib->GINV[0].compute_area(1)) + (req_width * FinFET_lib->GDFF[0].compute_area(1))) * 1.3  
												* router_info.n_out * router_info.n_v_channel * router_info.n_v_class;
					break;

				case QUEUE_ARBITER:
					/*
					router_area.vc_allocator = AreaDFF * router_info.vc_out_arb_queue_info.data_width 
						* router_info.vc_out_arb_queue_info.n_set * router_info.n_out * router_info.n_v_channel * router_info.n_v_class;
					*/
					router_area.vc_allocator = FinFET_lib->GDFF[0].compute_area(1) * router_info.vc_out_arb_queue_info.data_width 
						* router_info.vc_out_arb_queue_info.n_set * router_info.n_out * router_info.n_v_channel * router_info.n_v_class;

					break;

				default: printf ("error\n");  /* some error handler */
			}
		}

	}
	else if(router_info.vc_allocator_type == TWO_STAGE_ARB && router_info.n_v_channel > 1 && router_info.n_in > 1)
	{
		if (router_info.vc_in_arb_model && router_info.vc_out_arb_model){
			/*first stage*/
			req_width = router_info.n_v_channel;
			switch (router_info.vc_in_arb_model) {
				case MATRIX_ARBITER: //assumes 30% spacing for each arbiter
					/*
					router_area.vc_allocator = ((AreaNOR * 2 * (req_width - 1) * req_width) + (AreaINV * req_width)
							+ (AreaDFF * (req_width * (req_width - 1)/2))) * 1.3 * router_info.n_in * router_info.n_v_channel * router_info.n_v_class;
					*/
					router_area.vc_allocator = ((FinFET_lib->GNOR2[0].compute_area(1) * 2 * (req_width - 1) * req_width) + (FinFET_lib->GINV[0].compute_area(1) * req_width)
							+ (FinFET_lib->GDFF[0].compute_area(1) * (req_width * (req_width - 1)/2))) * 1.3 * router_info.n_in * router_info.n_v_channel * router_info.n_v_class;
					break;

				case RR_ARBITER: //assumes 30% spacing for each arbiter
					/*
					router_area.vc_allocator = ((6 *req_width * AreaNOR) + (2 * req_width * AreaINV) + (req_width * AreaDFF)) * 1.3 
										* router_info.n_in * router_info.n_v_channel * router_info.n_v_class ;
					*/
					router_area.vc_allocator = ((6 *req_width * FinFET_lib->GNOR2[0].compute_area(1)) + (2 * req_width * FinFET_lib->GINV[0].compute_area(1)) + (req_width * FinFET_lib->GDFF[0].compute_area(1))) * 1.3 
										* router_info.n_in * router_info.n_v_channel * router_info.n_v_class ;
					break;

				case QUEUE_ARBITER:
					/*
					router_area.vc_allocator = AreaDFF * router_info.vc_in_arb_queue_info.data_width
						* router_info.vc_in_arb_queue_info.n_set * router_info.n_in * router_info.n_v_channel * router_info.n_v_class ; 
					*/
					router_area.vc_allocator = FinFET_lib->GDFF[0].compute_area(1) * router_info.vc_in_arb_queue_info.data_width
						* router_info.vc_in_arb_queue_info.n_set * router_info.n_in * router_info.n_v_channel * router_info.n_v_class ; 

					break;

				default: printf ("error\n");  /* some error handler */
			}

			/*second stage*/
			req_width = (router_info.n_in - 1) * router_info.n_v_channel;
			switch (router_info.vc_out_arb_model) {
				case MATRIX_ARBITER: //assumes 30% spacing for each arbiter
				/*
				router_area.vc_allocator += ((AreaNOR * 2 * (req_width - 1) * req_width) + (AreaINV * req_width)
						+ (AreaDFF * (req_width * (req_width - 1)/2))) * 1.3 * router_info.n_out * router_info.n_v_channel * router_info.n_v_class;
				*/
				router_area.vc_allocator += ((FinFET_lib->GNOR2[0].compute_area(1) * 2 * (req_width - 1) * req_width) + (FinFET_lib->GINV[0].compute_area(1) * req_width)
						+ (FinFET_lib->GDFF[0].compute_area(1) * (req_width * (req_width - 1)/2))) * 1.3 * router_info.n_out * router_info.n_v_channel * router_info.n_v_class;
				break;

				case RR_ARBITER: //assumes 30% spacing for each arbiter
				/*
				router_area.vc_allocator += ((6 *req_width * AreaNOR) + (2 * req_width * AreaINV) + (req_width * AreaDFF)) * 1.3
										* router_info.n_out * router_info.n_v_channel * router_info.n_v_class;
				*/
				router_area.vc_allocator += ((6 *req_width * FinFET_lib->GNOR2[0].compute_area(1)) + (2 * req_width * FinFET_lib->GINV[0].compute_area(1)) + (req_width * FinFET_lib->GDFF[0].compute_area(1))) * 1.3
										* router_info.n_out * router_info.n_v_channel * router_info.n_v_class;
				break;

				case QUEUE_ARBITER:
				/*
				router_area.vc_allocator += AreaDFF * router_info.vc_out_arb_queue_info.data_width
					* router_info.vc_out_arb_queue_info.n_set * router_info.n_out * router_info.n_v_channel * router_info.n_v_class;
				*/
				router_area.vc_allocator += FinFET_lib->GDFF[0].compute_area(1) * router_info.vc_out_arb_queue_info.data_width
					* router_info.vc_out_arb_queue_info.n_set * router_info.n_out * router_info.n_v_channel * router_info.n_v_class;
				break;

				default: printf ("error\n");  /* some error handler */
			}


		}
	}
	else if(router_info.vc_allocator_type == VC_SELECT && router_info.n_v_channel > 1) 
	{
		switch (router_info.vc_select_buf_type) {
			case SRAM:
				/*
				bitline_len = router_info.n_v_channel * (RegCellHeight + 2 * WordlineSpacing);
				wordline_len = SIM_logtwo(router_info.n_v_channel) * (RegCellWidth + 2 * (router_info.vc_select_buf_info.read_ports
							+ router_info.vc_select_buf_info.write_ports) * BitlineSpacing);
				router_area.vc_allocator = router_info.n_out * router_info.n_v_class * (bitline_len * wordline_len);
				*/
				bitline_len = router_info.n_v_channel * (FinFET_lib->MemCell.Height);
				wordline_len = SIM_logtwo(router_info.n_v_channel) * (FinFET_lib->MemCell.Width);
				router_area.vc_allocator = router_info.n_out * router_info.n_v_class * (bitline_len * wordline_len);

				break;

			case REGISTER:
				/*
				router_area.vc_allocator = AreaDFF * router_info.n_out * router_info.n_v_class* router_info.vc_select_buf_info.data_width
					* router_info.vc_select_buf_info.n_set;
				*/
				router_area.vc_allocator = FinFET_lib->GDFF[0].compute_area(1) * router_info.n_out * router_info.n_v_class* router_info.vc_select_buf_info.data_width
					* router_info.vc_select_buf_info.n_set;

				break;

			default: printf ("error\n");  /* some error handler */

		}
	}


}

double ORION_Router ::  get_router_area()
{
	double Atotal;
	Atotal = router_area.buffer + router_area.crossbar + router_area.vc_allocator + router_area.sw_allocator;

#if( PARM(TECH_POINT) <= 90 )
	fprintf(stdout, "Abuffer:%g\t ACrossbar:%g\t AVCAllocator:%g\t ASWAllocator:%g\t Atotal:%g\n", router_area.buffer, router_area.crossbar, router_area.vc_allocator, router_area.sw_allocator,  Atotal);	

#else
    fprintf(stderr, "Router area is only supported for 90nm, 65nm, 45nm and 32nm\n");
#endif
	return Atotal;
}