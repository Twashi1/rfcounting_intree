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

#ifndef _SIM_ROUTER_H
#define _SIM_ROUTER_H

#include "SIM_parameter.h"
#include "SIM_array.h"
#include "SIM_crossbar.h"
#include "SIM_arbiter.h"
#include "SIM_clock.h"
#include "UCA.h"

using namespace std;

// Define the router area components
typedef struct {
	double buffer;
	double crossbar;
	double vc_allocator;
	double sw_allocator;
} SIM_router_area_t;

// Define the router configurations.  TODO: We can combine it with InputParameters in the future.
typedef struct {
	int n_in;
	int n_cache_in;
	int n_mc_in;
	int n_io_in;
	int n_out;
	int n_cache_out;
	int n_mc_out;
	int n_io_out;
	int flit_width;
	// virtual channel parameters
	int n_v_channel;
	int n_v_class;
	int cache_class;
	int mc_class;
	int io_class;
	int in_share_buf;	// whether input virtual classes share buffer
	int out_share_buf;	// whether output virtual classes share buffer
	int in_share_switch;	// whether input virtual classes share switch
	int out_share_switch;	// whether output virtual classes share switch
	// crossbar parameters
	int crossbar_model;
	int degree;	// only used by multree crossbar
	int connect_type;
	int trans_type;	// only used by transmission gate connection
	int xb_in_seg;	// only used by segmented crossbar
	int xb_out_seg; // only used by segmented crossbar
	double crossbar_in_len;
	double crossbar_out_len;
	// HACK HACK HACK
	int exp_xb_model;
	int exp_in_seg;	// only used by segmented crossbar
	int exp_out_seg;	// only used by segmented crossbar
	// buffer parameters
	int in_buf;
	int cache_in_buf;
	int mc_in_buf;
	int io_in_buf;
	int out_buf;
	int in_buffer_model;
	int out_buffer_model;
	// assume no buffering for local output ports
	int central_buf;
	SIM_array_info_t in_buf_info;
	SIM_array_info_t cache_in_buf_info;
	SIM_array_info_t mc_in_buf_info;
	SIM_array_info_t io_in_buf_info;
	SIM_array_info_t out_buf_info;
	SIM_array_info_t central_buf_info;
	int pipe_depth;
	// FIXME: this should be put in SIM_array_info_t
	int cbuf_ff_model;
	// switch allocator arbiter parameters
	int sw_in_arb_model;
	int sw_out_arb_model;
	int sw_in_arb_ff_model;
	int sw_out_arb_ff_model;
	// virtual channel allocator arbiter parameters
	int vc_allocator_type;
	int vc_in_arb_model;
	int vc_out_arb_model;
	int vc_in_arb_ff_model;
	int vc_out_arb_ff_model;
	int vc_select_buf_type;
	SIM_array_info_t sw_in_arb_queue_info;
	SIM_array_info_t cache_in_arb_queue_info;
	SIM_array_info_t mc_in_arb_queue_info;
	SIM_array_info_t io_in_arb_queue_info;
	SIM_array_info_t sw_out_arb_queue_info;
	SIM_array_info_t vc_in_arb_queue_info;
	SIM_array_info_t vc_out_arb_queue_info;
	SIM_array_info_t vc_select_buf_info;
	// clock related parameters
	int pipeline_stages;	
	int H_tree_clock;
	double router_diagonal;
	// redundant fields
	int n_total_in;
	int n_total_out;
	int in_n_switch;
	int cache_n_switch;
	int mc_n_switch;
	int io_n_switch;
	int n_switch_in;
	int n_switch_out;
} SIM_router_info_t;

// Define the router components
typedef struct {
	SIM_crossbar_t crossbar;
	SIM_crossbar_t exp_xb;
	SIM_array_t in_buf;
	SIM_array_t cache_in_buf;
	SIM_array_t mc_in_buf;
	SIM_array_t io_in_buf;
	SIM_array_t out_buf;
	SIM_array_t central_buf;
	SIM_crossbar_t in_cbuf_crsbar;
	SIM_crossbar_t out_cbuf_crsbar;
	SIM_arbiter_t cache_in_arb;
	SIM_arbiter_t mc_in_arb;
	SIM_arbiter_t io_in_arb;
	SIM_arbiter_t vc_in_arb;
	SIM_arbiter_t vc_out_arb;
	SIM_arbiter_t sw_in_arb;
	SIM_arbiter_t sw_out_arb;
	SIM_array_t vc_select_buf;
	/* FIXME: this should be put in SIM_array_t */
	SIM_ff_t cbuf_ff;
	double I_static;
	double I_buf_static;
	double I_crossbar_static;
	double I_vc_arbiter_static;
	double I_sw_arbiter_static;
	double I_clock_static;

	// Newly defined objects.  The previous objects are to be deleted.
	InputParameter* IP;
	TechnologyParameter* TP;
	DynamicParameter* DP;
	FinFET_Design_Library* FinFET_lib;
	PV_parameters PVP;

	// TODO: Have a Mat element to replace those SIM_arrays

	// Input buffer
	UCA * m_uca;

	// Crossbar
	Crossbar m_crossbar;
	Crossbar m_exp_xb;
	Crossbar m_in_cbuf_crsbar;
	Crossbar m_out_cbuf_crsbar;

	// Arbiter
	Arbiter m_cache_in_arb;
	Arbiter m_mc_in_arb;
	Arbiter m_io_in_arb;
	Arbiter m_vc_in_arb;
	Arbiter m_vc_out_arb;
	Arbiter m_sw_in_arb;
	Arbiter m_sw_out_arb;

	// Clock
	Clock m_CLK;

	// Leak Macros
	Leak_Macro Buffers_Leak;
	Leak_Macro Arbiters_Leak;
	Leak_Macro Router_Leak;

} SIM_router_power_t;

//interface for McPAT-PVT
class ORION_Router: public Component 
{
	public:
	ORION_Router();
	~ORION_Router();

	SIM_router_area_t router_area;
	SIM_router_info_t router_info;
	SIM_router_power_t router_power;
	InputParameter *IP;
	TechnologyParameter *TP;
	DynamicParameter *DP;
	FinFET_Design_Library *FinFET_lib;
	PV_parameters PVP;

	void router_power_init();//compute the leakage power of router's components and also router
	void router_area_init();//compute the area of router's components
	double get_router_area();//compute the total area of router

};
/* global variables */
extern GLOBDEF(SIM_router_power_t, router_power);
extern GLOBDEF(SIM_router_info_t, router_info);
extern GLOBDEF(SIM_router_area_t, router_area);


extern int SIM_router_init(SIM_router_info_t *info, SIM_router_power_t *router_power, SIM_router_area_t *router_area);

extern int SIM_buf_power_data_read(SIM_array_info_t *info, SIM_array_t *arr, LIB_Type_max_uint data);
extern int SIM_buf_power_data_write(SIM_array_info_t *info, SIM_array_t *arr, u_char *data_line, u_char *old_data, u_char *new_data);

extern int SIM_router_power_init(SIM_router_info_t *info, SIM_router_power_t *router);

extern int SIM_router_power_report(SIM_router_info_t *info, SIM_router_power_t *router);
extern double SIM_router_stat_energy(SIM_router_info_t *info, SIM_router_power_t *router, int print_depth, char *path, int max_avg, double e_fin, int plot_flag, double freq);

extern int SIM_router_area_init(SIM_router_info_t *info, SIM_router_area_t *router_area);
extern double SIM_router_area(SIM_router_area_t *router_area);

extern int SIM_crossbar_record(SIM_crossbar_t *xb, int io, LIB_Type_max_uint new_data, LIB_Type_max_uint old_data, int new_port, int old_port);
extern int SIM_arbiter_record(SIM_arbiter_t *arb, LIB_Type_max_uint new_req, LIB_Type_max_uint old_req, int new_grant, int old_grant);


#endif /* _SIM_ROUTER_H */

