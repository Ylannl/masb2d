import numpy as np
from pyflann import FLANN

import math

from algebra import cos_angle, projfac, compute_radius

# FIXME: can't handle duplicate points in input

class mapper(object):
    def __init__(self, function_pts):
        self.fx = [float(p[0]) for p in function_pts]
        self.fy = [float(p[1]) for p in function_pts]

    def f(self,x):
        i = None
        for i, fx in enumerate(self.fx):
            if x >= fx:
                pass
            else:
                break
        yi = self.fy[i]
        yi_ = self.fy[i-1]
        xi = self.fx[i]
        xi_ = self.fx[i-1]
        # print i, yi, yi_, xi, xi_
        if yi == yi_: return yi_
        return yi_ + ((yi - yi_)/(xi - xi_))*(x-xi_)

class MA(object):

    def __init__(self, datadict, maxR, denoise_absmin=None, denoise_delta=None, denoise_min=None, detect_planar=None):
        self.D = datadict # dict of numpy arrays
        # self.kd_tree = KDTree(self.D['coords'])

        # linear algorithm means brute force, which means its exact nn, which we need
        # approximate nn may cause algorithm not to converge
        self.flann = FLANN()
        self.flann.build_index(self.D['coords'], algorithm='linear',target_precision=1, sample_fraction=0.001,  log_level = "info")
        # print "constructed kd-tree"
        self.m, self.n = datadict['coords'].shape
        self.D['ma_coords_in'] = np.empty( (self.m,self.n) )
        self.D['ma_coords_in'][:] = np.nan
        self.D['ma_coords_out'] = np.empty( (self.m,self.n) )
        self.D['ma_coords_out'][:] = np.nan
        self.D['ma_radii_in'] = np.empty( (self.m) )
        self.D['ma_radii_in'][:] = np.nan
        self.D['ma_radii_out'] = np.empty( (self.m) )
        self.D['ma_radii_out'][:] = np.nan
        self.D['ma_f1_in'] = np.zeros( (self.m), dtype=np.int )
        self.D['ma_f1_in'][:] = np.nan
        self.D['ma_f1_out'] = np.zeros( (self.m), dtype=np.int )
        self.D['ma_f1_out'][:] = np.nan
        self.D['ma_f2_in'] = np.zeros( (self.m), dtype=np.int  )
        self.D['ma_f2_in'][:] = np.nan
        self.D['ma_f2_out'] = np.zeros( (self.m), dtype=np.int  )
        self.D['ma_f2_out'][:] = np.nan

        # a list of lists with indices of closest points during the ball shrinking process for every point:
        self.D['ma_shrinkhist_in'] = []
        self.D['ma_shrinkhist_out'] = []

        self.SuperR = maxR

        if denoise_absmin is None:
            self.denoise_absmin = None
        else:
            self.denoise_absmin = (math.pi/180)*denoise_absmin
        if denoise_delta is None:
            self.denoise_delta = None
        else:
            self.denoise_delta = (math.pi/180)*denoise_delta
        if denoise_min is None:
            self.denoise_min = None
        else:
            self.denoise_min = (math.pi/180)*denoise_min

        if detect_planar is None:
            self.detect_planar = None
        else:
            self.detect_planar = (math.pi/180)*detect_planar
        # self.normal_thres = 0.99

    def compute_balls_inout(self):
        for stage in self.compute_balls(inner=True):
            pass
        for stage in self.compute_balls(inner=False):
            pass

    def compute_lfs(self):
        self.ma_kd_tree = FLANN()

        # collect all ma_coords that are not NaN
        ma_coords = np.concatenate([self.D['ma_coords_in'], self.D['ma_coords_out']])
        ma_coords = ma_coords[~np.isnan(ma_coords).any(axis=1)]

        self.ma_kd_tree.build_index(ma_coords, algorithm='linear')
        # we can get *squared* distances for free, so take the square root
        self.D['lfs'] = np.sqrt(self.ma_kd_tree.nn_index(self.D['coords'], 1)[1])

    def decimate_lfs(self, m, scramble = False, sort = False):
        i=0
        self.D['decimate_lfs'] = np.zeros(self.m) == True

        plfs = zip(self.D['coords'], self.D['lfs'])
        if scramble: 
            from random import shuffle
            shuffle( plfs )
        if sort: 
            plfs.sort(key = lambda item: item[1])
            plfs.reverse()

        for p, lfs in plfs:
            if type(m) is float:
                qts = self.flann.nn_radius(p, (lfs*m)**2)[0][1:]
            else:
                qts = self.flann.nn_radius(p, m.f(lfs)**2)[0][1:]
            
            iqts = np.invert(self.D['decimate_lfs'][qts])
            if iqts.any():
                self.D['decimate_lfs'][i] = True
            i+=1

    def refine_lfs(self, m, scramble = False, sort = False):

        def brute_force_nn(q, coords):
            """return index of the closest point in coords"""
            distances = np.sqrt( np.square( coords[:,0]-q[0] ) + np.square( coords[:,1]-q[1] ) );

            return np.argsort(distances)[0]

        i=0
        self.D['decimate_lfs'] = np.zeros(self.m) == False

        plfs = zip(self.D['coords'], self.D['lfs'])
        if scramble: 
            from random import shuffle
            shuffle( plfs )
        if sort: 
            plfs.sort(key = lambda item: item[1])
            plfs.reverse()

        tmp_coords = np.array()

        for p, lfs in plfs:
            if type(m) is float:
                qts = self.flann.nn_radius(p, (lfs*m)**2)[0][1:]
            else:
                qts = self.flann.nn_radius(p, m.f(lfs)**2)[0][1:]
            
            iqts = np.invert(self.D['decimate_lfs'][qts])
            if iqts.any():
                self.D['decimate_lfs'][i] = True
            i+=1

    def compute_boundary_lenghts_2d(self):
        '''Compute for every point the boundary distance to the first point'''
        self.D['bound_len'] = np.zeros(self.m)
        i=1
        for p in self.D['coords'][1:]:
            self.D['bound_len'][i] = self.D['bound_len'][i-1] + np.linalg.norm(p-self.D['coords'][i-1])
            i+=1

    def compute_lam(self, inner='in'):
        '''Compute for every boundary point p, corresponding ma point m, and other feature point p_ the distance p-p_ '''
        self.D['lam_'+inner] = np.zeros(self.m)
        self.D['lam_'+inner][:] = np.nan

        for i, p in enumerate(self.D['coords']):
            c_p = self.D['ma_coords_'+inner][i]
            if not np.isnan(c_p[0]):
                p_ = self.D['coords'][self.D['ma_f2_'+inner][i]]
                self.D['lam_'+inner][i] = np.linalg.norm(p-p_)

    def compute_theta(self, inner='in'):
        '''Compute for every boundary point p, corresponding ma point m, and other feature point p_ the angle p-m-p_ '''
        self.D['theta_'+inner] = np.zeros(self.m)
        self.D['theta_'+inner][:] = np.nan

        for i, p in enumerate(self.D['coords']):
            c_p = self.D['ma_coords_'+inner][i]
            if not np.isnan(c_p[0]):
                p_ = self.D['coords'][self.D['ma_f2_'+inner][i]]
                self.D['theta_'+inner][i] = cos_angle(p-c_p, p_-c_p)

    def decimate_ballco(self, xi=0.1, k=4, inner='in'):
        self.D['decimate_ballco'] = np.zeros(self.m) == True

        for i, p in enumerate(self.D['coords']):
            c_p = self.D['ma_coords_'+inner][i]
            r_p = self.D['ma_radii_'+inner][i]
            if not np.isnan(c_p[0]):
                indices,dists = self.flann.nn_index(p, k+1)

                # convert indices to coordinates and radii
                M = [ (self.D['ma_coords_'+inner][index], self.D['ma_radii_'+inner][index]) for index in indices[0][1:] ]

                for m, r_m in M:
                    # can this medial ball (c_p) be contained by medial ball at m?
                    if np.linalg.norm(m-c_p) + r_p < r_m * (1+xi):
                        self.D['decimate_ballco'][i] = True
                        break

                # ballcos = [ r_m/np.linalg.norm(m-c_p) for m, r_m in M ]
                # self.D['ballco'][i] = max(ballcos)


    def decimate_heur(self, xi=0.1, k=3, omega=math.pi/20, inner='in'):
        '''Decimation based on heuristics as defined in ma (2012)'''
        cos_omega = math.cos(omega)
        self.D['filtered'] = np.zeros(self.m) == True
        
        for i, p in enumerate(self.D['coords']):
            c_p = self.D['ma_coords_'+inner][i]
            r_p = self.D['ma_radii_'+inner][i]
            if not np.isnan(c_p[0]):
                # test 1 - angle feature points
                p_ = self.D['coords'][self.D['ma_f2_'+inner][i]]
                if cos_angle(p, c_p, p_) < cos_omega:
                    self.D['filtered'][i] = True
                    break

                # test 2 - ball containmment
                indices,dists = self.flann.nn_index(p, k+1)

                M = [ ( self.D['ma_coords_'+inner][index], self.D['ma_radii_'+inner][index] ) for index in indices[0][1:] ]

                for m, r_m in M:
                    # can this medial ball (c_p) be contained by medial ball at m?
                    if np.linalg.norm(m-c_p) + r_p < r_m * (1+xi):
                        self.D['filtered'][i] = True
                        break
                

    def filter_radiuscon(self, alpha, k, inner='in'):
        '''Filter noisy points based on contuity in radius when compared to near points'''
        self.D['filter_radiuscon'] = np.zeros(self.m) == True
        
        for i, p in enumerate(self.D['coords']):
            c_p = self.D['ma_coords_'+inner][i]
            r_p = self.D['ma_radii_'+inner][i]
            if c_p != None:
                indices,dists = self.flann.nn_index(p, k+1)

                # print indices,dists
                M = []
                for index in indices[0][1:]:
                    M.append(self.D['ma_coords_'+inner][index])
                # print M

                L = []
                for m in M:
                    # projection_len = np.linalg.norm(proj(m-p,c_p-p))
                    val = np.linalg.norm(p-m) * cos_angle(m-p, c_p-p)
                    L.append(val)
                # print L, alpha * max(L), r_p

                if r_p < alpha * max(L):
                    self.D['filter_radiuscon'][i] = True
                else:
                    self.D['filter_radiuscon'][i] = False

    def filter_thetacon(self, theta_min=37, theta_delta=45, theta_absmin=26, inner='in'):
        """Filter noisy points based on continuity in separation angle as function of the ith iteration in the shrinking ball process"""
        # TODO: points with k=1 now receive no filtering... just discard them?
        self.D['filter_thetacon'] = np.zeros(self.m) == True

        theta_min *= (math.pi/180)
        theta_delta *= (math.pi/180)
        theta_absmin *= (math.pi/180)

        def find_optimal_theta(thetas):
            theta_prev = thetas[0]
            for j, theta in enumerate(thetas[1:]):
                if ( (theta_prev - theta) >= theta_delta and theta <= theta_min ) or (theta < theta_absmin):
                    return j
                theta_prev = theta
            # print
            return None

        for i, p in enumerate(self.D['coords']):
            p_n = self.D['normals'][i]

            q_indices = self.D['ma_shrinkhist_'+inner][i]
            if len(q_indices) <= 1: continue

            q_coords = self.D['coords'][q_indices]
            
            # if not is_inner: p_n = -p_n

            radii = [ compute_radius(p,p_n,q) for q in q_coords ]
            centers = [ p - p_n * r for r in radii ]
            thetas = [ math.acos(cos_angle(p-c,q-c)) for c, q in zip(centers, q_coords) ]

            optimal_theta = find_optimal_theta(thetas)
            # print optimal_theta
            if optimal_theta is not None:
                self.D['filter_thetacon'][i] = True

    def compute_balls(self, inner=True, verbose=False):
        """Balls shrinking algorithm. Set `inner` to False when outer balls are wanted."""

        for i, pn in enumerate(zip(self.D['coords'], self.D['normals'])):
            p, n = pn
            if not inner:
                n = -n
            
            # when approximating 1st point initialize q with random point not equal to p
            q=p 
            # if i==0:
            #     while (q == p).all():
            #         random_index = int(rand(1)*self.D['coords'].shape[0])
            #         q = self.D['coords'][random_index]
            #     r = compute_radius(p,n,q)

            # forget optimization of r:
            r=self.SuperR
            
            msg='New iteration, initial r = {:.5}'.format(float(r))
            if verbose: print msg
            yield {'stage': 1, 'geom': (p,n), 'msg':msg}

            r_ = None
            c = None
            j = -1
            q_i = None
            q_history = []
            while True:
                j+=1
                # initialize r on last found radius
                if j>0:
                    r = r_
                elif j==0 and i>0:
                    r = r

                # compute ball center
                c = p - n*r
                #
                q_i_previous = q_i
                
                msg = 'Current iteration: #' + str(i) +', r = {:.5}'.format(float(r))
                if verbose: print msg
                yield {'stage': 2, 'geom': (q,c,r), 'msg':msg}

                ### FINDING NEAREST NEIGHBOR OF c

                # find closest point to c and assign to q
                indices,dists = self.flann.nn_index(c, 2)
                # dists, indices = self.kd_tree.query(array([c]), k=2)
                candidate_c = self.D['coords'][indices]
                # candidate_n= self.D['normals'][indices]
                # print 'candidates:', candidates
                q = candidate_c[0][0]
                # q_n = candidate_n[0][0]
                q_i = indices[0][0]
                
                # yield {'stage': 3, 'geom': (q)}

                # What to do if closest point is p itself?
                if (q==p).all():
                    # 1) if r==SuperR, apparantly no other points on the halfspace spanned by -n => that's an infinite ball
                    if r == self.SuperR: 
                        r_ = r
                        break
                    # 2) otherwise just pick the second closest point
                    else: 
                        q = candidate_c[0][1]
                        # q_n = candidate_n[0][1]
                        q_i = indices[0][1]
                
                q_history.append(q_i)
                # compute new candidate radius r_
                r_ = compute_radius(p,n,q)

                # print r, r_, p-c, q-c, cos_angle(p-c, q-c)

                ### BOUNDARY CASES

                # if r_ < 0 closest point was on the wrong side of plane with normal n => start over with SuperRadius on the right side of that plance
                if r_ < 0: 
                    r_ = self.SuperR
                # if r_ > SuperR, stop now because otherwise in case of planar surface point configuration, we end up in an infinite loop
                elif r_ > self.SuperR:
                # elif cos_angle(p-c, q-c) >= self.normal_thres:
                    r_ = self.SuperR
                    break

                c_ = p - n*r_
                # this seems to work well against noisy ma points.
                if self.denoise_absmin is not None:
                    if math.acos(cos_angle(p-c_, q-c_)) < self.denoise_absmin and j>0 and r_>np.linalg.norm(q-p):
                        # msg = 'Current iteration: -#' + str(i) +', r = {:.5}'.format(float(r))
                        # yield {'stage': 2, 'geom': (q,c_,r), 'msg':msg}
                        # keep previous radius:
                        r_=r
                        q_i = q_i_previous
                        break

                if self.denoise_delta is not None and j>0:
                    theta_now = math.acos(cos_angle(p-c_, q-c_))
                    q_previous = self.D['coords'][q_i_previous]
                    theta_prev = math.acos(cos_angle(p-c_, q_previous-c_))
                    
                    if theta_prev-theta_now > self.denoise_delta and theta_now < self.denoise_min and r_>np.linalg.norm(q-p):
                        # print "theta_prev:",theta_prev/math.pi * 180
                        # print "theta_now:",theta_now/math.pi * 180
                        # print "self.denoise_delta:",self.denoise_delta/math.pi * 180
                        # print "self.denoise_min:",self.denoise_min/math.pi * 180

                        # keep previous radius:
                        r_=r
                        q_i = q_i_previous
                        break

                if self.detect_planar != None:
                    if math.acos( cos_angle(q-p, -n) ) > self.detect_planar and j<2:
                        # yield {'stage': 2, 'geom': (q,p - n*r_,r_), 'msg':msg}
                        r_= self.SuperR
                        # r_= r
                        # q_i = q_i_previous
                        break

                ### NORMAL STOP CONDITION

                # stop iteration if r has converged
                if r == r_:
                    break
            
            if inner: inout = 'in'
            else: inout = 'out'
            
            if r_ >= self.SuperR:
                pass
            else:
                self.D['ma_radii_'+inout][i] = r_
                self.D['ma_coords_'+inout][i] = c
                self.D['ma_f1_'+inout][i] = i
                self.D['ma_f2_'+inout][i] = q_i
            self.D['ma_shrinkhist_'+inout].append(q_history[:-1])

    def construct_topo_2d(self, inner='in', project=True):

        def arrayindex(A, value):
            tmp = np.where(A==value)
            # print tmp, tmp[0].shape
            if tmp[0].shape != (0,): return tmp[0][0]
            else: return np.nan

        self.D['ma_linepieces_'+inner] = list()
        if project:
            for index in xrange(1,self.m):
                index_1 = index-1
                
                # find ma points corresponding to these three feature points
                f2_p = arrayindex(self.D['ma_f2_'+inner], index_1)
                f2 = arrayindex(self.D['ma_f2_'+inner], index)
                f1_p = arrayindex(self.D['ma_f1_'+inner], index_1)
                f1 = arrayindex(self.D['ma_f1_'+inner], index)

                # collect unique id's of corresponding ma_coords
                S = set()
                for f in [f1,f1_p, f2, f2_p]:
                    if not np.isnan(f):
                        S.add( f )

                # this is the linevector we are projecting the ma_coords on:
                l = self.D['coords'][index] - self.D['coords'][index_1]

                # compute projections of ma_coords on line l
                S_ = list()
                for s in S:
                    # if not np.isnan(self.D['ma_coords_'+inner][s]):
                    S_.append( (projfac(l, self.D['ma_coords_'+inner][s]-self.D['coords'][index_1] ), s) )

                # now we can sort them on their x coordinate
                S_.sort(key=lambda item: item[0])

                # now we have the line segments
                for i in xrange(len(S_)):
                    self.D['ma_linepieces_'+inner].append( (S_[i-1][1], S_[i][1]) )
        else:
            indices = list()
            for i in xrange(self.m):
                if not np.isnan(self.D['ma_coords_'+inner][i][0]):
                    indices.append(i)

            for i in xrange(1,len(indices)):
                s = indices[i-1]
                e = indices[i]
                self.D['ma_linepieces_'+inner].append((s,e))
            # connect last point to first
            # self.D['ma_linepieces_'+inner].append((indices[-1], indices[0]))
            