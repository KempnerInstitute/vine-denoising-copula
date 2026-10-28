// Optional CPU execution of the corrected, linear HFuncLookup D-vine sampler.
// Compile without fast-math or contraction: arithmetic order matches NumPy.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

static inline double clip(double x, double lo=0., double hi=1.) {
    return std::min(std::max(x, lo), hi);
}
static inline double inv_value(int index, int m, const double* h, bool transpose,
                               int left, int right, double f) {
    if (index == 0) return 0.;
    if (index >= m+1) return 1.;
    int t = index-1;
    double a = transpose ? h[left*m+t] : h[t*m+left];
    double b = transpose ? h[right*m+t] : h[t*m+right];
    return clip((1.-f)*a+f*b);
}
static inline double hinv(double q, double other, const double* nodes, int m,
                          const double* h, bool transpose) {
    int right = std::lower_bound(nodes, nodes+m, other)-nodes;
    right = std::min(std::max(right,1),m-1);
    int left = right-1;
    double f = clip((other-nodes[left])/(nodes[right]-nodes[left]));
    int low=0, high=m+2;
    while (low<high) {
        int mid=(low+high)/2;
        if (inv_value(mid,m,h,transpose,left,right,f)<q) low=mid+1;
        else high=mid;
    }
    if (low==0) return 0.;
    int upper=std::min(std::max(low,1),m+1);
    double h0=inv_value(upper-1,m,h,transpose,left,right,f);
    double h1=inv_value(upper,m,h,transpose,left,right,f);
    double width=h1-h0;
    double fq=std::abs(width)>=1e-12 ? (q-h0)/width : 0.;
    double x0=upper==1 ? 0. : nodes[upper-2];
    double x1=upper==m+1 ? 1. : nodes[upper-1];
    return clip(x0+fq*(x1-x0));
}
static inline double bilinear(double u, double v, const double* nodes, int m,
                              const double* h) {
    double du=nodes[1]-nodes[0];
    double iu=(clip(u,nodes[0],nodes[m-1])-nodes[0])/du;
    double iv=(clip(v,nodes[0],nodes[m-1])-nodes[0])/du;
    int i=std::min(std::max(int(std::floor(iu)),0),m-2);
    int j=std::min(std::max(int(std::floor(iv)),0),m-2);
    double tu=clip(iu-i),tv=clip(iv-j);
    return clip((1.-tu)*(1.-tv)*h[i*m+j]+tu*(1.-tv)*h[(i+1)*m+j]
                 +(1.-tu)*tv*h[i*m+j+1]+tu*tv*h[(i+1)*m+j+1]);
}
static inline double forward_h(double target, double other, const double* nodes,
                               int m, const double* h, bool target_first) {
    double base=target_first ? bilinear(target,other,nodes,m,h)
                             : bilinear(other,target,nodes,m,h);
    if (target<nodes[0]) base=(clip(target,0.,nodes[0])/nodes[0])*base;
    if (target>nodes[m-1])
        base=base+(clip(target,nodes[m-1],1.)-nodes[m-1])*(1.-base)/(1.-nodes[m-1]);
    return clip(base);
}

extern "C" int vdc_dvine_inverse(const double* w, double* out, std::int64_t n,
    int d, int m, const std::int64_t* order, const std::int64_t* orientation,
    const double* nodes, const double* tables, int threads, int block_size) {
    // Each block owns its sample conditionals; no random generation or reductions.
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::int64_t start=0;start<n;start+=block_size) {
        int b=std::min(std::int64_t(block_size),n-start);
        std::vector<double> c(d*b),after(d*b),q(b);
        for (int r=0;r<b;++r) out[(start+r)*d+order[0]]=c[r]=clip(w[(start+r)*d+order[0]],1e-6,1.-1e-6);
        for (int k=1;k<d;++k) {
            for (int r=0;r<b;++r) q[r]=clip(w[(start+r)*d+order[k]],1e-6,1.-1e-6);
            for (int i=0;i<k;++i) {
                auto edge=std::int64_t(k)*(k-1)/2+i;
                auto ori=orientation[edge];
                const double* h=tables+edge*2*m*m+(ori==1 ? m*m : 0);
                for (int r=0;r<b;++r) {
                    if (ori) q[r]=clip(hinv(q[r],c[i*b+r],nodes,m,h,ori==1),1e-6,1.-1e-6);
                    after[i*b+r]=q[r];
                }
            }
            for (int r=0;r<b;++r) out[(start+r)*d+order[k]]=c[k*b+r]=q[r];
            for (int i=k-1;i>=0;--i) {
                auto edge=std::int64_t(k)*(k-1)/2+i;
                auto ori=orientation[edge];
                const double* h=tables+edge*2*m*m+(ori==1 ? 0 : m*m);
                if (ori) for (int r=0;r<b;++r)
                    c[i*b+r]=clip(forward_h(c[i*b+r],after[i*b+r],nodes,m,h,ori==1),1e-6,1.-1e-6);
            }
        }
    }
    return 0;
}
