import{c as e,l as t,u as n}from"./index-B1KNpdRP.js";var r=1e-7,i=1e-4,a=class{constructor(e,t){this.backend=e,this.dataMover=t,this.data=new WeakMap,this.dataIdsCount=0}get(e){return this.data.has(e)||this.dataMover.moveData(this.backend,e),this.data.get(e)}set(e,t){this.dataIdsCount++,this.data.set(e,t)}has(e){return this.data.has(e)}delete(e){return this.dataIdsCount--,this.data.delete(e)}numDataIds(){return this.dataIdsCount}},o=class{refCount(e){return s(`refCount`)}incRef(e){return s(`incRef`)}timerAvailable(){return!0}time(e){return s(`time`)}read(e){return s(`read`)}readSync(e){return s(`readSync`)}readToGPU(e,t){return s(`readToGPU`)}numDataIds(){return s(`numDataIds`)}disposeData(e,t){return s(`disposeData`)}write(e,t,n){return s(`write`)}move(e,t,n,r,i){return s(`move`)}memory(){return s(`memory`)}floatPrecision(){return s(`floatPrecision`)}epsilon(){return this.floatPrecision()===32?r:i}dispose(){return s(`dispose`)}};function s(e){throw Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}function c(e,t,n){return Math.max(e,Math.min(t,n))}function l(e){return e%2==0?e:e+1}function u(e,t,n){let r=e[t];e[t]=e[n],e[n]=r}function d(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}function f(e,t){if(!e)throw Error(typeof t==`string`?t:t())}function p(e,t,n=``){f(_(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function m(e){f(e!=null,()=>`The input to the tensor constructor must be a non-null value.`)}function h(e,t=[],n=!1){if(t??=[],Array.isArray(e)||k(e)&&!n)for(let r=0;r<e.length;++r)h(e[r],t,n);else t.push(e);return t}function g(e){if(e.length===0)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function _(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function v(e){return e%1==0}function y(e){let t=Math.ceil(Math.sqrt(e));return[t,Math.ceil(e/t)]}function b(e,t){return t<=e.length?e:e+` `.repeat(t-e.length)}function x(e,t=e=>0,n,r=setTimeout){return new Promise((i,a)=>{let o=0,s=()=>{if(e()){i();return}o++;let c=t(o);n!=null&&o>=n?a():r(s,c)};s()})}function S(e,t){let n=1,r=-1;for(let t=0;t<e.length;++t)if(e[t]>=0)n*=e[t];else if(e[t]===-1){if(r!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${r} and dim ${t}`);r=t}else if(e[t]<0)throw Error(`Shapes can not be < 0. Found ${e[t]} at dim ${t}`);if(r===-1){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(n===0)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);let i=e.slice();return i[r]=t/n,i}function C(e,t){let n=t.length;return e=e==null?t.map((e,t)=>t):[].concat(e),f(e.every(e=>e>=-n&&e<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),f(e.every(e=>v(e)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(e=>e<0?n+e:e)}function w(e,t){let n=[],r=[],i=t!=null&&Array.isArray(t)&&t.length===0,a=t==null||i?null:C(t,e).sort(),o=0;for(let t=0;t<e.length;++t){if(a!=null){if(a[o]===t&&e[t]!==1)throw Error(`Can't squeeze axis ${t} since its dim '${e[t]}' is not 1`);(a[o]==null||a[o]>t)&&e[t]===1&&(n.push(e[t]),r.push(t)),a[o]<=t&&o++}e[t]!==1&&(n.push(e[t]),r.push(t))}return{newShape:n,keptDims:r}}function T(e,t){let n=null;if(e==null||e===`float32`)n=new Float32Array(t);else if(e===`int32`)n=new Int32Array(t);else if(e===`bool`)n=new Uint8Array(t);else throw Error(`Unknown data type ${e}`);return n}function E(e,t){let n=null;if(e==null||e===`float32`)n=new Float32Array(t);else if(e===`int32`)n=new Int32Array(t);else if(e===`bool`)n=new Uint8Array(t);else if(e===`string`)n=Array(t);else throw Error(`Unknown data type ${e}`);return n}function D(e,t){for(let n=0;n<e.length;n++){let r=e[n];if(isNaN(r)||!isFinite(r))throw Error(`A tensor of type ${t} being uploaded contains ${r}.`)}}function O(e){return e===`bool`||e===`complex64`||e===`float32`||e===`int32`||e===`string`}function ee(e,t){return!(t===`complex64`||t===`float32`&&e!==`complex64`||t===`int32`&&e!==`float32`&&e!==`complex64`||t===`bool`&&e===`bool`)}function k(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}function A(e){if(e===`float32`||e===`int32`)return 4;if(e===`complex64`)return 8;if(e===`bool`)return 1;throw Error(`Unknown dtype ${e}`)}function te(e){if(e==null)return 0;let t=0;return e.forEach(e=>t+=e.length),t}function ne(e){return typeof e==`string`||e instanceof String}function re(e){return typeof e==`boolean`}function ie(e){return typeof e==`number`}function ae(e){return Array.isArray(e)?ae(e[0]):e instanceof Float32Array?`float32`:e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?`int32`:ie(e)?`float32`:ne(e)?`string`:re(e)?`bool`:`float32`}function oe(e){return!!(e&&e.constructor&&e.call&&e.apply)}function se(e,t){for(let n=t;n<e;++n)if(e%n===0)return n;return e}function j(e){let t=e.length;if(t<2)return[];let n=Array(t-1);n[t-2]=e[t-1];for(let r=t-3;r>=0;--r)n[r]=n[r+1]*e[r+1];return n}function ce(e,t,n,r=!1){let i=[];if(t.length===1){let a=t[0]*(r?2:1);for(let t=0;t<a;t++)i[t]=n[e+t]}else{let a=t[0],o=t.slice(1),s=o.reduce((e,t)=>e*t)*(r?2:1);for(let t=0;t<a;t++)i[t]=ce(e+t*s,o,n,r)}return i}function le(e,t,n=!1){if(e.length===0)return t[0];let r=e.reduce((e,t)=>e*t)*(n?2:1);if(r===0)return[];if(r!==t.length)throw Error(`[${e}] does not match the input size ${t.length}${n?` for a complex tensor`:``}.`);return ce(0,e,t,n)}function ue(e,t){let n=M(e,t);for(let e=0;e<n.length;e++)n[e]=1;return n}function M(e,t){if(t==null||t===`float32`||t===`complex64`)return new Float32Array(e);if(t===`int32`)return new Int32Array(e);if(t===`bool`)return new Uint8Array(e);throw Error(`Unknown data type ${t}`)}function de(e,t){let n=e.reduce((e,t)=>e*t,1);if(t==null||t===`float32`)return le(e,new Float32Array(n));if(t===`int32`)return le(e,new Int32Array(n));if(t===`bool`)return le(e,new Uint8Array(n));throw Error(`Unknown data type ${t}`)}function fe(e){e.forEach(t=>{f(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function pe(e,t,n){if(t===0)return 0;if(t===1)return e[0];let r=e[e.length-1];for(let t=0;t<e.length-1;++t)r+=n[t]*e[t];return r}function me(e,t,n){if(t===0)return[];if(t===1)return[e];let r=Array(t);for(let t=0;t<r.length-1;++t)r[t]=Math.floor(e/n[t]),e-=r[t]*n[t];return r[r.length-1]=e,r}function he(e){return e&&e.then&&typeof e.then==`function`}var ge=`tfjsflags`,_e=class{constructor(e){this.global=e,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=ve,this.populateURLFlags()}setPlatform(e,t){this.platform!=null&&(N().getBool(`IS_TEST`)||N().getBool(`PROD`)||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${e}.`)),this.platformName=e,this.platform=t}registerFlag(e,t,n){if(this.flagRegistry[e]={evaluationFn:t,setHook:n},this.urlFlags[e]!=null){let t=this.urlFlags[e];N().getBool(`IS_TEST`)||N().getBool(`PROD`)||console.warn(`Setting feature override from URL ${e}: ${t}.`),this.set(e,t)}}async getAsync(e){return e in this.flags||(this.flags[e]=await this.evaluateFlag(e)),this.flags[e]}get(e){if(e in this.flags)return this.flags[e];let t=this.evaluateFlag(e);if(he(t))throw Error(`Flag ${e} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[e]=t,this.flags[e]}getNumber(e){return this.get(e)}getBool(e){return this.get(e)}getFlags(){return this.flags}get features(){return this.flags}set(e,t){if(this.flagRegistry[e]==null)throw Error(`Cannot set flag ${e} as it has not been registered.`);this.flags[e]=t,this.flagRegistry[e].setHook!=null&&this.flagRegistry[e].setHook(t)}evaluateFlag(e){if(this.flagRegistry[e]==null)throw Error(`Cannot evaluate flag '${e}': no evaluation function found.`);return this.flagRegistry[e].evaluationFn()}setFlags(e){this.flags=Object.assign({},e)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(this.global===void 0||this.global.location===void 0||this.global.location.search===void 0)return;let e=this.getQueryParams(this.global.location.search);ge in e&&e[ge].split(`,`).forEach(e=>{let[t,n]=e.split(`:`);this.urlFlags[t]=be(t,n)})}};function ve(e){let t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(e,...n)=>(ye(t,n[0],n[1]),n.join(`=`))),t}function ye(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||``)}function be(e,t){if(t=t.toLowerCase(),t===`true`||t===`false`)return t===`true`;if(`${+t}`===t)return+t;throw Error(`Could not parse value flag value ${t} for flag ${e}.`)}function N(){return xe}var xe=null;function Se(e){xe=e}var Ce;function we(){if(Ce==null){let e;if(typeof window<`u`)e=window;else if(typeof global<`u`)e=global;else if(typeof process<`u`)e=process;else if(typeof self<`u`)e=self;else throw Error(`Could not find a global object`);Ce=e}return Ce}function Te(){let e=we();return e._tfGlobals??=new Map,e._tfGlobals}function Ee(e,t){let n=Te();if(n.has(e))return n.get(e);{let r=t();return n.set(e,r),n.get(e)}}var De=`Acos`,Oe=`Acosh`,ke=`AddN`,Ae=`ArgMax`,je=`ArgMin`,Me=`Asin`,Ne=`Asinh`,Pe=`Atan`,Fe=`Atanh`,Ie=`Atan2`,Le=`AvgPool`,Re=`AvgPoolGrad`,ze=`AvgPool3D`,Be=`AvgPool3DGrad`,Ve=`BatchMatMul`,He=`BatchToSpaceND`,Ue=`Bincount`,We=`BroadcastArgs`,Ge=`Cast`,Ke=`Ceil`,qe=`ClipByValue`,Je=`Complex`,Ye=`ComplexAbs`,Xe=`Concat`,Ze=`Conv2D`,Qe=`Conv2DBackpropFilter`,$e=`Conv2DBackpropInput`,et=`Conv3D`,tt=`Conv3DBackpropFilterV2`,nt=`Conv3DBackpropInputV2`,rt=`Cosh`,it=`Cumprod`,at=`Cumsum`,ot=`CropAndResize`,st=`DenseBincount`,ct=`DepthToSpace`,lt=`DepthwiseConv2dNative`,ut=`DepthwiseConv2dNativeBackpropFilter`,dt=`DepthwiseConv2dNativeBackpropInput`,ft=`Diag`,pt=`Dilation2D`,mt=`Dilation2DBackpropInput`,ht=`Dilation2DBackpropFilter`,gt=`RealDiv`,_t=`Einsum`,vt=`EluGrad`,yt=`Equal`,bt=`ExpandDims`,xt=`Expm1`,St=`Fill`,Ct=`FlipLeftRight`,wt=`Floor`,Tt=`FloorDiv`,Et=`FusedBatchNorm`,Dt=`GatherV2`,Ot=`GatherNd`,kt=`Greater`,At=`GreaterEqual`,jt=`Identity`,Mt=`IFFT`,Nt=`Imag`,Pt=`IsFinite`,Ft=`IsInf`,It=`IsNan`,Lt=`LeakyRelu`,Rt=`Less`,zt=`LessEqual`,Bt=`LinSpace`,Vt=`Log1p`,Ht=`LogicalAnd`,Ut=`LogicalNot`,Wt=`LogicalOr`,Gt=`LRNGrad`,Kt=`Maximum`,qt=`MaxPool`,Jt=`MaxPoolGrad`,Yt=`MaxPool3D`,Xt=`MaxPool3DGrad`,Zt=`MaxPoolWithArgmax`,Qt=`Mean`,$t=`Minimum`,en=`MirrorPad`,tn=`Multinomial`,nn=`Multiply`,rn=`NotEqual`,an=`NonMaxSuppressionV3`,on=`NonMaxSuppressionV4`,sn=`NonMaxSuppressionV5`,cn=`OnesLike`,ln=`OneHot`,un=`Pack`,dn=`PadV2`,fn=`Prelu`,pn=`Prod`,mn=`RaggedGather`,hn=`RaggedTensorToTensor`,gn=`Range`,_n=`Real`,vn=`Reciprocal`,yn=`Relu`,bn=`Reshape`,xn=`ResizeNearestNeighbor`,Sn=`ResizeNearestNeighborGrad`,Cn=`ResizeBilinear`,wn=`ResizeBilinearGrad`,Tn=`Relu6`,En=`Reverse`,Dn=`Round`,On=`Rsqrt`,kn=`ScatterNd`,An=`SearchSorted`,jn=`Select`,Mn=`Selu`,Nn=`Slice`,Pn=`Sinh`,Fn=`Sign`,In=`Sigmoid`,Ln=`Softplus`,Rn=`Sqrt`,zn=`SpaceToBatchND`,Bn=`SplitV`,Vn=`Softmax`,Hn=`SparseFillEmptyRows`,Un=`SparseReshape`,Wn=`SparseSegmentMean`,Gn=`SparseSegmentSum`,Kn=`SparseToDense`,qn=`SquaredDifference`,Jn=`Square`,Yn=`StridedSlice`,Xn=`StringNGrams`,Zn=`StringSplit`,Qn=`StringToHashBucketFast`,$n=`Tanh`,er=`Tile`,tr=`TopK`,nr=`Transform`,rr=`Transpose`,ir=`Unique`,ar=`Unpack`,or=`UnsortedSegmentSum`,sr=`ZerosLike`,cr=`Step`,lr=`FromPixels`,ur=`RotateWithOffset`,dr=`_FusedMatMul`,fr=`FusedConv2D`,pr=`FusedDepthwiseConv2D`;function mr(...e){N().getBool(`IS_TEST`)||N().getBool(`PROD`)||console.warn(...e)}function hr(...e){N().getBool(`IS_TEST`)||N().getBool(`PROD`)||console.log(...e)}var gr=Ee(`kernelRegistry`,()=>new Map),_r=Ee(`gradRegistry`,()=>new Map);function vr(e,t){let n=Sr(e,t);return gr.get(n)}function yr(e){return _r.get(e)}function br(e){let t=gr.entries(),n=[];for(;;){let{done:r,value:i}=t.next();if(r)break;let[a,o]=i,[s]=a.split(`_`);s===e&&n.push(o)}return n}function xr(e){let{kernelName:t,backendName:n}=e,r=Sr(t,n);gr.has(r)&&mr(`The kernel '${t}' for backend '${n}' is already registered`),gr.set(r,e)}function Sr(e,t){return`${t}_${e}`}var Cr=n(e(((e,t)=>{t.exports=r;var n=null;try{n=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function r(e,t,n){this.low=e|0,this.high=t|0,this.unsigned=!!n}r.prototype.__isLong__,Object.defineProperty(r.prototype,"__isLong__",{value:!0});function i(e){return(e&&e.__isLong__)===!0}r.isLong=i;var a={},o={};function s(e,t){var n,r,i;return t?(e>>>=0,(i=0<=e&&e<256)&&(r=o[e],r)?r:(n=l(e,(e|0)<0?-1:0,!0),i&&(o[e]=n),n)):(e|=0,(i=-128<=e&&e<128)&&(r=a[e],r)?r:(n=l(e,e<0?-1:0,!1),i&&(a[e]=n),n))}r.fromInt=s;function c(e,t){if(isNaN(e))return t?b:y;if(t){if(e<0)return b;if(e>=g)return T}else{if(e<=-_)return E;if(e+1>=_)return w}return e<0?c(-e,t).neg():l(e%h|0,e/h|0,t)}r.fromNumber=c;function l(e,t,n){return new r(e,t,n)}r.fromBits=l;var u=Math.pow;function d(e,t,n){if(e.length===0)throw Error(`empty string`);if(e===`NaN`||e===`Infinity`||e===`+Infinity`||e===`-Infinity`)return y;if(typeof t==`number`?(n=t,t=!1):t=!!t,n||=10,n<2||36<n)throw RangeError(`radix`);var r;if((r=e.indexOf(`-`))>0)throw Error(`interior hyphen`);if(r===0)return d(e.substring(1),t,n).neg();for(var i=c(u(n,8)),a=y,o=0;o<e.length;o+=8){var s=Math.min(8,e.length-o),l=parseInt(e.substring(o,o+s),n);if(s<8){var f=c(u(n,s));a=a.mul(f).add(c(l))}else a=a.mul(i),a=a.add(c(l))}return a.unsigned=t,a}r.fromString=d;function f(e,t){return typeof e==`number`?c(e,t):typeof e==`string`?d(e,t):l(e.low,e.high,typeof t==`boolean`?t:e.unsigned)}r.fromValue=f;var p=65536,m=1<<24,h=p*p,g=h*h,_=g/2,v=s(m),y=s(0);r.ZERO=y;var b=s(0,!0);r.UZERO=b;var x=s(1);r.ONE=x;var S=s(1,!0);r.UONE=S;var C=s(-1);r.NEG_ONE=C;var w=l(-1,2147483647,!1);r.MAX_VALUE=w;var T=l(-1,-1,!0);r.MAX_UNSIGNED_VALUE=T;var E=l(0,-2147483648,!1);r.MIN_VALUE=E;var D=r.prototype;D.toInt=function(){return this.unsigned?this.low>>>0:this.low},D.toNumber=function(){return this.unsigned?(this.high>>>0)*h+(this.low>>>0):this.high*h+(this.low>>>0)},D.toString=function(e){if(e||=10,e<2||36<e)throw RangeError(`radix`);if(this.isZero())return`0`;if(this.isNegative()){if(this.eq(E)){var t=c(e),n=this.div(t),r=n.mul(t).sub(this);return n.toString(e)+r.toInt().toString(e)}return`-`+this.neg().toString(e)}for(var i=c(u(e,6),this.unsigned),a=this,o=``;;){var s=a.div(i),l=(a.sub(s.mul(i)).toInt()>>>0).toString(e);if(a=s,a.isZero())return l+o;for(;l.length<6;)l=`0`+l;o=``+l+o}},D.getHighBits=function(){return this.high},D.getHighBitsUnsigned=function(){return this.high>>>0},D.getLowBits=function(){return this.low},D.getLowBitsUnsigned=function(){return this.low>>>0},D.getNumBitsAbs=function(){if(this.isNegative())return this.eq(E)?64:this.neg().getNumBitsAbs();for(var e=this.high==0?this.low:this.high,t=31;t>0&&!(e&1<<t);t--);return this.high==0?t+1:t+33},D.isZero=function(){return this.high===0&&this.low===0},D.eqz=D.isZero,D.isNegative=function(){return!this.unsigned&&this.high<0},D.isPositive=function(){return this.unsigned||this.high>=0},D.isOdd=function(){return(this.low&1)==1},D.isEven=function(){return!(this.low&1)},D.equals=function(e){return i(e)||(e=f(e)),this.unsigned!==e.unsigned&&this.high>>>31==1&&e.high>>>31==1?!1:this.high===e.high&&this.low===e.low},D.eq=D.equals,D.notEquals=function(e){return!this.eq(e)},D.neq=D.notEquals,D.ne=D.notEquals,D.lessThan=function(e){return this.comp(e)<0},D.lt=D.lessThan,D.lessThanOrEqual=function(e){return this.comp(e)<=0},D.lte=D.lessThanOrEqual,D.le=D.lessThanOrEqual,D.greaterThan=function(e){return this.comp(e)>0},D.gt=D.greaterThan,D.greaterThanOrEqual=function(e){return this.comp(e)>=0},D.gte=D.greaterThanOrEqual,D.ge=D.greaterThanOrEqual,D.compare=function(e){if(i(e)||(e=f(e)),this.eq(e))return 0;var t=this.isNegative(),n=e.isNegative();return t&&!n?-1:!t&&n?1:this.unsigned?e.high>>>0>this.high>>>0||e.high===this.high&&e.low>>>0>this.low>>>0?-1:1:this.sub(e).isNegative()?-1:1},D.comp=D.compare,D.negate=function(){return!this.unsigned&&this.eq(E)?E:this.not().add(x)},D.neg=D.negate,D.add=function(e){i(e)||(e=f(e));var t=this.high>>>16,n=this.high&65535,r=this.low>>>16,a=this.low&65535,o=e.high>>>16,s=e.high&65535,c=e.low>>>16,u=e.low&65535,d=0,p=0,m=0,h=0;return h+=a+u,m+=h>>>16,h&=65535,m+=r+c,p+=m>>>16,m&=65535,p+=n+s,d+=p>>>16,p&=65535,d+=t+o,d&=65535,l(m<<16|h,d<<16|p,this.unsigned)},D.subtract=function(e){return i(e)||(e=f(e)),this.add(e.neg())},D.sub=D.subtract,D.multiply=function(e){if(this.isZero())return y;if(i(e)||(e=f(e)),n)return l(n.mul(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned);if(e.isZero())return y;if(this.eq(E))return e.isOdd()?E:y;if(e.eq(E))return this.isOdd()?E:y;if(this.isNegative())return e.isNegative()?this.neg().mul(e.neg()):this.neg().mul(e).neg();if(e.isNegative())return this.mul(e.neg()).neg();if(this.lt(v)&&e.lt(v))return c(this.toNumber()*e.toNumber(),this.unsigned);var t=this.high>>>16,r=this.high&65535,a=this.low>>>16,o=this.low&65535,s=e.high>>>16,u=e.high&65535,d=e.low>>>16,p=e.low&65535,m=0,h=0,g=0,_=0;return _+=o*p,g+=_>>>16,_&=65535,g+=a*p,h+=g>>>16,g&=65535,g+=o*d,h+=g>>>16,g&=65535,h+=r*p,m+=h>>>16,h&=65535,h+=a*d,m+=h>>>16,h&=65535,h+=o*u,m+=h>>>16,h&=65535,m+=t*p+r*d+a*u+o*s,m&=65535,l(g<<16|_,m<<16|h,this.unsigned)},D.mul=D.multiply,D.divide=function(e){if(i(e)||(e=f(e)),e.isZero())throw Error(`division by zero`);if(n)return!this.unsigned&&this.high===-2147483648&&e.low===-1&&e.high===-1?this:l((this.unsigned?n.div_u:n.div_s)(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned);if(this.isZero())return this.unsigned?b:y;var t,r,a;if(this.unsigned){if(e.unsigned||(e=e.toUnsigned()),e.gt(this))return b;if(e.gt(this.shru(1)))return S;a=b}else{if(this.eq(E))return e.eq(x)||e.eq(C)?E:e.eq(E)?x:(t=this.shr(1).div(e).shl(1),t.eq(y)?e.isNegative()?x:C:(r=this.sub(e.mul(t)),a=t.add(r.div(e)),a));if(e.eq(E))return this.unsigned?b:y;if(this.isNegative())return e.isNegative()?this.neg().div(e.neg()):this.neg().div(e).neg();if(e.isNegative())return this.div(e.neg()).neg();a=y}for(r=this;r.gte(e);){t=Math.max(1,Math.floor(r.toNumber()/e.toNumber()));for(var o=Math.ceil(Math.log(t)/Math.LN2),s=o<=48?1:u(2,o-48),d=c(t),p=d.mul(e);p.isNegative()||p.gt(r);)t-=s,d=c(t,this.unsigned),p=d.mul(e);d.isZero()&&(d=x),a=a.add(d),r=r.sub(p)}return a},D.div=D.divide,D.modulo=function(e){return i(e)||(e=f(e)),n?l((this.unsigned?n.rem_u:n.rem_s)(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned):this.sub(this.div(e).mul(e))},D.mod=D.modulo,D.rem=D.modulo,D.not=function(){return l(~this.low,~this.high,this.unsigned)},D.and=function(e){return i(e)||(e=f(e)),l(this.low&e.low,this.high&e.high,this.unsigned)},D.or=function(e){return i(e)||(e=f(e)),l(this.low|e.low,this.high|e.high,this.unsigned)},D.xor=function(e){return i(e)||(e=f(e)),l(this.low^e.low,this.high^e.high,this.unsigned)},D.shiftLeft=function(e){return i(e)&&(e=e.toInt()),(e&=63)==0?this:e<32?l(this.low<<e,this.high<<e|this.low>>>32-e,this.unsigned):l(0,this.low<<e-32,this.unsigned)},D.shl=D.shiftLeft,D.shiftRight=function(e){return i(e)&&(e=e.toInt()),(e&=63)==0?this:e<32?l(this.low>>>e|this.high<<32-e,this.high>>e,this.unsigned):l(this.high>>e-32,this.high>=0?0:-1,this.unsigned)},D.shr=D.shiftRight,D.shiftRightUnsigned=function(e){if(i(e)&&(e=e.toInt()),e&=63,e===0)return this;var t=this.high;if(e<32){var n=this.low;return l(n>>>e|t<<32-e,t>>>e,this.unsigned)}return l(e===32?t:t>>>e-32,0,this.unsigned)},D.shru=D.shiftRightUnsigned,D.shr_u=D.shiftRightUnsigned,D.toSigned=function(){return this.unsigned?l(this.low,this.high,!1):this},D.toUnsigned=function(){return this.unsigned?this:l(this.low,this.high,!0)},D.toBytes=function(e){return e?this.toBytesLE():this.toBytesBE()},D.toBytesLE=function(){var e=this.high,t=this.low;return[t&255,t>>>8&255,t>>>16&255,t>>>24,e&255,e>>>8&255,e>>>16&255,e>>>24]},D.toBytesBE=function(){var e=this.high,t=this.low;return[e>>>24,e>>>16&255,e>>>8&255,e&255,t>>>24,t>>>16&255,t>>>8&255,t&255]},r.fromBytes=function(e,t,n){return n?r.fromBytesLE(e,t):r.fromBytesBE(e,t)},r.fromBytesLE=function(e,t){return new r(e[0]|e[1]<<8|e[2]<<16|e[3]<<24,e[4]|e[5]<<8|e[6]<<16|e[7]<<24,t)},r.fromBytesBE=function(e,t){return new r(e[4]<<24|e[5]<<16|e[6]<<8|e[7],e[0]<<24|e[1]<<16|e[2]<<8|e[3],t)}}))()),wr=Cr.default||Cr;function Tr(e){return wr.fromString(e,!0,16)}var Er=Tr(`c3a5c85c97cb3127`),Dr=Tr(`b492b66fbe98f273`),Or=Tr(`9ae16a3b2f90404f`);function kr(e){return e.xor(e.shru(47))}function Ar(e,t,n){let r=e.slice(t,t+n);return wr.fromBytes(Array.from(r),!0,!0)}function jr(e,t){return Ar(e,t,8)}function Mr(e,t){return Ar(e,t,4)}function Nr(e,t){return t===0?e:e.shru(t).or(e.shl(64-t))}function Pr(e,t,n=Tr(`9ddfea08eb382d69`)){let r=e.xor(t).mul(n);r=r.xor(r.shru(47));let i=t.xor(r).mul(n);return i=i.xor(i.shru(47)),i=i.mul(n),i}function Fr(e,t,n,r,i,a){i=i.add(e),a=Nr(a.add(i).add(r),21);let o=i;return i=i.add(t),i=i.add(n),a=a.add(Nr(i,44)),[i.add(r),a.add(o)]}function Ir(e,t,n,r){return Fr(jr(e,t),jr(e,t+8),jr(e,t+16),jr(e,t+24),n,r)}function Lr(e,t=e.length){if(t>=8){let n=Or.add(t*2),r=jr(e,0).add(Or),i=jr(e,t-8);return Pr(Nr(i,37).mul(n).add(r),Nr(r,25).add(i).mul(n),n)}if(t>=4){let n=Or.add(t*2);return Pr(Mr(e,0).shl(3).add(t),Mr(e,t-4),n)}if(t>0){let n=e[0],r=e[t>>1],i=e[t-1],a=n+(r<<8),o=t+(i<<2);return kr(Or.mul(a).xor(Er.mul(o))).mul(Or)}return Or}function Rr(e,t=e.length){let n=Or.add(t*2),r=jr(e,0).mul(Dr),i=jr(e,8),a=jr(e,t-8).mul(n),o=jr(e,t-16).mul(Or);return Pr(Nr(r.add(i),43).add(Nr(a,30)).add(o),r.add(Nr(i.add(Or),18)).add(a),n)}function zr(e,t=e.length){let n=Or.add(t*2),r=jr(e,0).mul(Or),i=jr(e,8),a=jr(e,t-8).mul(n),o=jr(e,t-16).mul(Or),s=Nr(r.add(i),43).add(Nr(a,30)).add(o),c=Pr(s,r.add(Nr(i.add(Or),18)).add(a),n),l=jr(e,16).mul(n),u=jr(e,24),d=s.add(jr(e,t-32)).mul(n),f=c.add(jr(e,t-24)).mul(n);return Pr(Nr(l.add(u),43).add(Nr(d,30)).add(f),l.add(Nr(u.add(r),18)).add(d),n)}function Br(e,t=e.length){let n=wr.fromNumber(81,!0);if(t<=32)return t<=16?Lr(e,t):Rr(e,t);if(t<=64)return zr(e,t);let r=n,i=n.mul(Dr).add(113),a=kr(i.mul(Or).add(113)).mul(Or),o=[wr.UZERO,wr.UZERO],s=[wr.UZERO,wr.UZERO];r=r.mul(Or).add(jr(e,0));let c=0,l=(t-1>>6)*64,u=l+(t-1&63)-63;do r=Nr(r.add(i).add(o[0]).add(jr(e,c+8)),37).mul(Dr),i=Nr(i.add(o[1]).add(jr(e,c+48)),42).mul(Dr),r=r.xor(s[1]),i=i.add(o[0]).add(jr(e,c+40)),a=Nr(a.add(s[0]),33).mul(Dr),o=Ir(e,c,o[1].mul(Dr),r.add(s[0])),s=Ir(e,c+32,a.add(s[1]),i.add(jr(e,c+16))),[a,r]=[r,a],c+=64;while(c!==l);let d=Dr.add(a.and(255).shl(1));return c=u,s[0]=s[0].add(t-1&63),o[0]=o[0].add(s[0]),s[0]=s[0].add(o[0]),r=Nr(r.add(i).add(o[0]).add(jr(e,c+8)),37).mul(d),i=Nr(i.add(o[1]).add(jr(e,c+48)),42).mul(d),r=r.xor(s[1].mul(9)),i=i.add(o[0].mul(9).add(jr(e,c+40))),a=Nr(a.add(s[0]),33).mul(d),o=Ir(e,c,o[1].mul(d),r.add(s[0])),s=Ir(e,c+32,a.add(s[1]),i.add(jr(e,c+16))),[a,r]=[r,a],Pr(Pr(o[0],s[0],d).add(kr(i).mul(Er)).add(a),Pr(o[1],s[1],d).add(r),d)}function Vr(e,t){return t===`string`?Gr(e):Ur([e],t)}function Hr(e,t){return e instanceof Float32Array&&t===`float32`||e instanceof Int32Array&&t===`int32`||e instanceof Uint8Array&&t===`bool`}function Ur(e,t){if(t===`string`)throw Error(`Cannot convert a string[] to a TypedArray`);if(Array.isArray(e)&&(e=h(e)),N().getBool(`DEBUG`)&&D(e,t),Hr(e,t))return e;if(t==null||t===`float32`||t===`complex64`)return new Float32Array(e);if(t===`int32`)return new Int32Array(e);if(t===`bool`){let t=new Uint8Array(e.length);for(let n=0;n<t.length;++n)Math.round(e[n])!==0&&(t[n]=1);return t}throw Error(`Unknown data type ${t}`)}function Wr(){return N().platform.now()}function Gr(e,t=`utf-8`){return t||=`utf-8`,N().platform.encode(e,t)}function Kr(e,t=`utf-8`){return t||=`utf-8`,N().platform.decode(e,t)}var qr=class{constructor(e,t){this.backendTimer=e,this.logger=t,t??(this.logger=new Yr)}profileKernel(e,t,n){let r,i=()=>{r=n()},a,o=Wr();if(this.backendTimer.timerAvailable())a=this.backendTimer.time(i);else{i();for(let e of r)e.dataSync();a=Promise.resolve({kernelMs:Wr()-o})}if(N().getBool(`CHECK_COMPUTATION_FOR_ERRORS`))for(let t=0;t<r.length;t++){let n=r[t];n.data().then(t=>{Jr(t,n.dtype,e)})}return{kernelName:e,outputs:r,inputs:t,timeMs:a.then(e=>e.kernelMs),extraInfo:a.then(e=>e.getExtraProfileInfo==null?``:e.getExtraProfileInfo())}}logKernelProfile(e){let{kernelName:t,outputs:n,timeMs:r,inputs:i,extraInfo:a}=e;n.forEach(e=>{Promise.all([e.data(),r,a]).then(n=>{this.logger.logKernelProfile(t,e,n[0],n[1],i,n[2])})})}};function Jr(e,t,n){if(t!==`float32`)return!1;for(let t=0;t<e.length;t++){let r=e[t];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${n}'`),!0}return!1}var Yr=class{logKernelProfile(e,t,n,r,i,a){let o=typeof r==`number`?b(`${r}ms`,9):r.error,s=b(e,25),c=t.rank,l=t.size,u=b(t.shape.toString(),14),d=``;for(let e in i){let n=i[e];if(n!=null){let r=n.shape||t.shape,i=r.length;d+=`${e}: ${i}D ${i>0?r:``} `}}console.log(`%c${s}\t%c${o}\t%c${c}D ${u}\t%c${l}\t%c${d}\t%c${a}`,`font-weight:bold`,`color:red`,`color:blue`,`color: orange`,`color: green`,`color: steelblue`)}};function Xr(e,t,n){let r={},i={};for(let e=0;e<t.length;e++)r[t[e].id]=!0;for(let n=0;n<e.length;n++){let a=e[n],o=a.inputs;for(let e in o){let n=o[e],s=!1;for(let e=0;e<t.length;e++)if(r[n.id]){a.outputs.forEach(e=>r[e.id]=!0),s=!0,i[a.id]=!0;break}if(s)break}}let a={};a[n.id]=!0;let o={};for(let t=e.length-1;t>=0;t--){let n=e[t],r=n.inputs;for(let e=0;e<n.outputs.length;e++)if(a[n.outputs[e].id]){for(let e in r)a[r[e].id]=!0,o[n.id]=!0;break}}let s=[];for(let t=0;t<e.length;t++){let n=e[t];if(i[n.id]&&o[n.id]){let e={};for(let t in n.inputs){let i=n.inputs[t];r[i.id]&&(e[t]=i)}let t=Object.assign({},n);t.inputs=e,t.outputs=n.outputs,s.push(t)}}return s}function Zr(e,t,n,r){for(let i=t.length-1;i>=0;i--){let a=t[i],o=[];if(a.outputs.forEach(t=>{let n=e[t.id];n==null?o.push(null):o.push(n)}),a.gradient==null)throw Error(`Cannot compute gradient: gradient function not found for ${a.kernelName}.`);let s=a.gradient(o);for(let t in a.inputs){if(!(t in s))throw Error(`Cannot backprop through input ${t}. Available gradients found: ${Object.keys(s)}.`);let i=n(()=>s[t]());if(i.dtype!==`float32`)throw Error(`Error in gradient for op ${a.kernelName}. The gradient of input ${t} must have 'float32' dtype, but has '${i.dtype}'`);let o=a.inputs[t];if(!_(i.shape,o.shape))throw Error(`Error in gradient for op ${a.kernelName}. The gradient of input '${t}' has shape '${i.shape}', which does not match the shape of the input '${o.shape}'`);if(e[o.id]==null)e[o.id]=i;else{let t=e[o.id];e[o.id]=r(t,i),t.dispose()}}}}var Qr=20,$r=3,ei=7;function ti(e,t,n,r){let i=j(t),a=ni(e,t,n,i),o=t.length,s=ai(e,t,n,i,a),c=[`Tensor`];return r&&(c.push(`  dtype: ${n}`),c.push(`  rank: ${o}`),c.push(`  shape: [${t}]`),c.push(`  values:`)),c.push(s.map(e=>`    `+e).join(`
`)),c.join(`
`)}function ni(e,t,n,r){let i=g(t),a=r[r.length-1],o=Array(a).fill(0),s=t.length,c=n===`complex64`?oi(e):e;if(s>1)for(let e=0;e<i/a;e++){let t=e*a;for(let e=0;e<a;e++)o[e]=Math.max(o[e],ri(c[t+e],0,n).length)}return o}function ri(e,t,n){let r;return r=Array.isArray(e)?`${parseFloat(e[0].toFixed(ei))} + ${parseFloat(e[1].toFixed(ei))}j`:ne(e)?`'${e}'`:n===`bool`?ii(e):parseFloat(e.toFixed(ei)).toString(),b(r,t)}function ii(e){return e===0?`false`:`true`}function ai(e,t,n,r,i,a=!0){let o=n===`complex64`?2:1,s=t[0],c=t.length;if(c===0)return n===`complex64`?[ri(oi(e)[0],0,n)]:n===`bool`?[ii(e[0])]:[e[0].toString()];if(c===1){if(s>Qr){let t=$r*o,r=Array.from(e.slice(0,t)),a=Array.from(e.slice((s-$r)*o,s*o));return n===`complex64`&&(r=oi(r),a=oi(a)),[`[`+r.map((e,t)=>ri(e,i[t],n)).join(`, `)+`, ..., `+a.map((e,t)=>ri(e,i[s-$r+t],n)).join(`, `)+`]`]}return[`[`+(n===`complex64`?oi(e):Array.from(e)).map((e,t)=>ri(e,i[t],n)).join(`, `)+`]`]}let l=t.slice(1),u=r.slice(1),d=r[0]*o,f=[];if(s>Qr){for(let t=0;t<$r;t++){let r=t*d,a=r+d;f.push(...ai(e.slice(r,a),l,n,u,i,!1))}f.push(`...`);for(let t=s-$r;t<s;t++){let r=t*d,a=r+d;f.push(...ai(e.slice(r,a),l,n,u,i,t===s-1))}}else for(let t=0;t<s;t++){let r=t*d,a=r+d;f.push(...ai(e.slice(r,a),l,n,u,i,t===s-1))}let p=c===2?`,`:``;f[0]=`[`+f[0]+p;for(let e=1;e<f.length-1;e++)f[e]=` `+f[e]+p;let m=`,
`;for(let e=2;e<c;e++)m+=`
`;return f[f.length-1]=` `+f[f.length-1]+`]`+(a?``:m),f}function oi(e){let t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}var si=class{constructor(e,t,n){if(this.dtype=t,this.shape=e.slice(),this.size=g(e),n!=null){let e=n.length;f(e===this.size,()=>`Length of values '${e}' does not match the size inferred by the shape '${this.size}'.`)}if(t===`complex64`)throw Error(`complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).`);this.values=n||E(t,this.size),this.strides=j(e)}set(e,...t){t.length===0&&(t=[0]),f(t.length===this.rank,()=>`The number of provided coordinates (${t.length}) must match the rank (${this.rank})`);let n=this.locToIndex(t);this.values[n]=e}get(...e){e.length===0&&(e=[0]);let t=0;for(let n of e){if(n<0||n>=this.shape[t]){let t=`Requested out of range element at ${e}.   Buffer shape=${this.shape}`;throw Error(t)}t++}let n=e[e.length-1];for(let t=0;t<e.length-1;++t)n+=this.strides[t]*e[t];return this.values[n]}locToIndex(e){if(this.rank===0)return 0;if(this.rank===1)return e[0];let t=e[e.length-1];for(let n=0;n<e.length-1;++n)t+=this.strides[n]*e[n];return t}indexToLoc(e){if(this.rank===0)return[];if(this.rank===1)return[e];let t=Array(this.shape.length);for(let n=0;n<t.length-1;++n)t[n]=Math.floor(e/this.strides[n]),e-=t[n]*this.strides[n];return t[t.length-1]=e,t}get rank(){return this.shape.length}toTensor(){return ci().makeTensor(this.values,this.shape,this.dtype)}},ci=null,li=null;function ui(e){ci=e}function di(e){li=e}var fi=class{constructor(e,t,n,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=e.slice(),this.dtype=t||`float32`,this.size=g(e),this.strides=j(e),this.dataId=n,this.id=r,this.rankType=this.rank<5?this.rank.toString():`higher`}get rank(){return this.shape.length}async buffer(){let e=await this.data();return li.buffer(this.shape,this.dtype,e)}bufferSync(){return li.buffer(this.shape,this.dtype,this.dataSync())}async array(){let e=await this.data();return le(this.shape,e,this.dtype===`complex64`)}arraySync(){return le(this.shape,this.dataSync(),this.dtype===`complex64`)}async data(){this.throwIfDisposed();let e=ci().read(this.dataId);if(this.dtype===`string`){let t=await e;try{return t.map(e=>Kr(e))}catch{throw Error(`Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().`)}}return e}dataToGPU(e){return this.throwIfDisposed(),ci().readToGPU(this.dataId,e)}dataSync(){this.throwIfDisposed();let e=ci().readSync(this.dataId);if(this.dtype===`string`)try{return e.map(e=>Kr(e))}catch{throw Error(`Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().`)}return e}async bytes(){this.throwIfDisposed();let e=await ci().read(this.dataId);return this.dtype===`string`?e:new Uint8Array(e.buffer)}dispose(){this.isDisposed||(ci().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw Error(`Tensor is disposed.`)}print(e=!1){return li.print(this,e)}clone(){return this.throwIfDisposed(),li.clone(this)}toString(e=!1){return ti(this.dataSync(),this.shape,this.dtype,e)}cast(e){return this.throwIfDisposed(),li.cast(this,e)}variable(e=!0,t,n){return this.throwIfDisposed(),ci().makeVariable(this,e,t,n)}};Object.defineProperty(fi,Symbol.hasInstance,{value:e=>!!e&&e.data!=null&&e.dataSync!=null&&e.throwIfDisposed!=null});function pi(){return Ee(`Tensor`,()=>fi)}pi();var mi=class extends fi{constructor(e,t,n,r){super(e.shape,e.dtype,e.dataId,r),this.trainable=t,this.name=n}assign(e){if(e.dtype!==this.dtype)throw Error(`dtype of the new value (${e.dtype}) and previous value (${this.dtype}) must match`);if(!_(e.shape,this.shape))throw Error(`shape of the new value (${e.shape}) and previous value (${this.shape}) must match`);ci().disposeTensor(this),this.dataId=e.dataId,ci().incRef(this,null)}dispose(){ci().disposeVariable(this),this.isDisposedInternal=!0}};Object.defineProperty(mi,Symbol.hasInstance,{value:e=>e instanceof fi&&e.assign!=null&&e.assign instanceof Function});var hi;(function(e){e.R0=`R0`,e.R1=`R1`,e.R2=`R2`,e.R3=`R3`,e.R4=`R4`,e.R5=`R5`,e.R6=`R6`})(hi||={});var gi;(function(e){e.float32=`float32`,e.int32=`int32`,e.bool=`int32`,e.complex64=`complex64`})(gi||={});var _i;(function(e){e.float32=`float32`,e.int32=`int32`,e.bool=`bool`,e.complex64=`complex64`})(_i||={});var vi;(function(e){e.float32=`float32`,e.int32=`float32`,e.bool=`float32`,e.complex64=`complex64`})(vi||={});var yi;(function(e){e.float32=`complex64`,e.int32=`complex64`,e.bool=`complex64`,e.complex64=`complex64`})(yi||={});var bi={float32:vi,int32:gi,bool:_i,complex64:yi};function xi(e,t){if(e===`string`||t===`string`){if(e===`string`&&t===`string`)return`string`;throw Error(`Can not upcast ${e} with ${t}`)}return bi[e][t]}function Si(e){return xi(e,`int32`)}function Ci(e,t){if(e.dtype===t.dtype)return[e,t];let n=xi(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function wi(e,t){f(e.dtype===t.dtype,()=>`The dtypes of the first(${e.dtype}) and second(${t.dtype}) input must match`)}function Ti(e){let t=[];return Ei(e,t,new Set),t}function Ei(e,t,n){if(e==null)return;if(e instanceof fi){t.push(e);return}if(!Di(e))return;let r=e;for(let e in r){let i=r[e];n.has(i)||(n.add(i),Ei(i,t,n))}}function Di(e){return Array.isArray(e)||typeof e==`object`}function Oi(e){return e.kernelName!=null}var ki=class{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(e=>e.name)))}}}dispose(){for(let e in this.registeredVariables)this.registeredVariables[e].dispose()}},Ai=class e{constructor(e){this.ENV=e,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new ki}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;let e=this.getSortedBackends();for(let t=0;t<e.length;t++){let n=e[t];if(await this.initializeBackend(n).success){await this.setBackend(n);return}}throw Error(`Could not initialize any backends, all backend initializations failed.`)}get backend(){if(this.pendingBackendInit!=null)throw Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){let{name:e,asyncInit:t}=this.initializeBackendsAndReturnBest();if(t)throw Error(`The highest priority backend '${e}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(e)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(e){if(!(e in this.registry)){if(e in this.registryFactory){let{asyncInit:t}=this.initializeBackend(e);if(t)return null}else return null}return this.registry[e]}findBackendFactory(e){return e in this.registryFactory?this.registryFactory[e].factory:null}registerBackend(e,t,n=1){return e in this.registryFactory?(mr(`${e} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[e]={factory:t,priority:n},!0)}async setBackend(e){if(this.registryFactory[e]==null)throw Error(`Backend name '${e}' not found in registry`);if(this.backendName=e,this.registry[e]==null){this.backendInstance=null;let{success:t,asyncInit:n}=this.initializeBackend(e);if(!(n?await t:t))return!1}return this.backendInstance=this.registry[e],this.setupRegisteredKernels(),this.profiler=new qr(this.backendInstance),!0}setupRegisteredKernels(){br(this.backendName).forEach(e=>{e.setupFunc!=null&&e.setupFunc(this.backendInstance)})}disposeRegisteredKernels(e){br(e).forEach(t=>{t.disposeFunc!=null&&t.disposeFunc(this.registry[e])})}initializeBackend(e){let t=this.registryFactory[e];if(t==null)throw Error(`Cannot initialize backend ${e}, no registration found.`);try{let n=t.factory();if(n&&!(n instanceof o)&&typeof n.then==`function`){let t=++this.pendingBackendInitId,r=n.then(n=>t<this.pendingBackendInitId?!1:(this.registry[e]=n,this.pendingBackendInit=null,!0)).catch(n=>(t<this.pendingBackendInitId||(this.pendingBackendInit=null,mr(`Initialization of backend ${e} failed`),mr(n.stack||n.message)),!1));return this.pendingBackendInit=r,{success:r,asyncInit:!0}}return this.registry[e]=n,{success:!0,asyncInit:!1}}catch(t){return mr(`Initialization of backend ${e} failed`),mr(t.stack||t.message),{success:!1,asyncInit:!1}}}removeBackend(e){if(!(e in this.registryFactory))throw Error(`${e} backend not found in registry`);this.backendName===e&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,e in this.registry&&(this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e]),delete this.registryFactory[e],this.backendName===e&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw Error(`No backend found in registry.`);return Object.keys(this.registryFactory).sort((e,t)=>this.registryFactory[t].priority-this.registryFactory[e].priority)}initializeBackendsAndReturnBest(){let e=this.getSortedBackends();for(let t=0;t<e.length;t++){let n=e[t],{success:r,asyncInit:i}=this.initializeBackend(n);if(i||r)return{name:n,asyncInit:i}}throw Error(`Could not initialize any backends, all backend initializations failed.`)}moveData(e,t){let n=this.state.tensorInfo.get(t),r=n.backend,i=this.readSync(t),a=r.refCount(t);r.disposeData(t,!0),n.backend=e,e.move(t,i,n.shape,n.dtype,a),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(e,t){let n=null;if(t==null){if(typeof e!=`function`)throw Error(`Please provide a function to tidy()`);t=e}else{if(typeof e!=`string`&&!(e instanceof String))throw Error(`When calling with two arguments, the first argument to tidy() must be a string`);if(typeof t!=`function`)throw Error(`When calling with two arguments, the 2nd argument to tidy() must be a function`);n=e}let r;return this.scopedRun(()=>this.startScope(n),()=>this.endScope(r),()=>(r=t(),r instanceof Promise&&console.error(`Cannot return a Promise inside of tidy.`),r))}scopedRun(e,t,n){e();try{let e=n();return t(),e}catch(e){throw t(),e}}nextTensorId(){return e.nextTensorId++}nextVariableId(){return e.nextVariableId++}clone(e){let t=P.runKernel(jt,{x:e}),n={x:e};return this.addTapeNode(this.state.activeScope.name,n,[t],e=>({x:()=>{let t={x:e};return P.runKernel(Ge,t,{dtype:`float32`})}}),[],{}),t}runKernel(e,t,n){if(this.backendName??this.backend,vr(e,this.backendName)==null)throw Error(`Kernel '${e}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:e,inputs:t,attrs:n})}shouldCheckForMemLeaks(){return this.ENV.getBool(`IS_TEST`)}checkKernelForMemLeak(e,t,n){let r=this.backend.numDataIds(),i=0;n.forEach(e=>{i+=e.dtype===`complex64`?3:1});let a=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],o=r-t-i-a;if(o>0)throw Error(`Backend '${this.backendName}' has an internal memory leak (${o} data ids) after running '${e}'`)}runKernelFunc(e){let t,n=[],r=this.isTapeOn(),i=this.state.numBytes,a=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let o;this.backendName??this.backend;let s,c=Oi(e)?e.kernelName:this.state.activeScope==null?``:this.state.activeScope.name;if(Oi(e)){let{kernelName:t,inputs:i,attrs:a}=e;this.backendName??this.backend;let c=vr(t,this.backendName);f(c!=null,()=>`Cannot find registered kernel '${t}' for backend '${this.backendName}'`),o=()=>{let e=this.backend.numDataIds();s=c.kernelFunc({inputs:i,attrs:a,backend:this.backend});let o=Array.isArray(s)?s:[s];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(t,e,o);let l=o.map(e=>e.rank==null?this.makeTensorFromTensorInfo(e):e);if(r){let e=this.getTensorsForGradient(t,i,l);n=this.saveTensorsForBackwardMode(e)}return l}}else{let{forwardFunc:t}=e,i=e=>{r&&(n=e.map(e=>this.keep(this.clone(e))))};o=()=>{let e=this.backend.numDataIds();s=this.tidy(()=>t(this.backend,i));let n=Array.isArray(s)?s:[s];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(c,e,n),n}}let{inputs:l,attrs:u}=e,d=Oi(e)?null:e.backwardsFunc,p;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool(`DEBUG`)&&!this.state.profiling?t=o():(p=this.profiler.profileKernel(c,l,()=>o()),this.ENV.getBool(`DEBUG`)&&this.profiler.logKernelProfile(p),t=p.outputs)}),r&&this.addTapeNode(c,l,t,d,n,u),this.state.profiling&&this.state.activeProfile.kernels.push({name:c,bytesAdded:this.state.numBytes-i,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-a,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(l).map(e=>l[e]==null?null:l[e].shape),outputShapes:t.map(e=>e.shape),kernelTimeMs:p.timeMs,extraInfo:p.extraInfo}),Array.isArray(s)?t:t[0]}saveTensorsForBackwardMode(e){return e.map(e=>this.keep(this.clone(e)))}getTensorsForGradient(e,t,n){let r=yr(e);if(r!=null){let e=r.inputsToSave||[],i=r.outputsToSave||[],a;r.saveAllInputs?(f(Array.isArray(t),()=>`saveAllInputs is true, expected inputs to be an array.`),a=Object.keys(t).map(e=>t[e])):a=e.map(e=>t[e]);let o=n.filter((e,t)=>i[t]);return a.concat(o)}return[]}makeTensor(e,t,n,r){if(e==null)throw Error(`Values passed to engine.makeTensor() are null`);n||=`float32`,r||=this.backend;let i=e;n===`string`&&ne(e[0])&&(i=e.map(e=>Gr(e)));let a=r.write(i,t,n),o=new fi(t,n,a,this.nextTensorId());if(this.trackTensor(o,r),n===`string`){let e=this.state.tensorInfo.get(a),t=te(i);this.state.numBytes+=t-e.bytes,e.bytes=t}return o}makeTensorFromDataId(e,t,n,r){n||=`float32`;let i={dataId:e,shape:t,dtype:n};return this.makeTensorFromTensorInfo(i,r)}makeTensorFromTensorInfo(e,t){let{dataId:n,shape:r,dtype:i}=e,a=new fi(r,i,n,this.nextTensorId());return this.trackTensor(a,t),a}makeVariable(e,t=!0,n,r){n||=this.nextVariableId().toString(),r!=null&&r!==e.dtype&&(e=e.cast(r));let i=new mi(e,t,n,this.nextTensorId());if(this.state.registeredVariables[i.name]!=null)throw Error(`Variable with name ${i.name} was already registered`);return this.state.registeredVariables[i.name]=i,this.incRef(i,this.backend),i}trackTensor(e,t){this.state.numTensors++,e.dtype===`string`&&this.state.numStringTensors++;let n=0;e.dtype!==`complex64`&&e.dtype!==`string`&&(n=e.size*A(e.dtype)),this.state.numBytes+=n,this.state.tensorInfo.has(e.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(e.dataId,{backend:t||this.backend,dtype:e.dtype,shape:e.shape,bytes:n})),e instanceof mi||this.track(e)}incRef(e,t){this.trackTensor(e,t),this.backend.incRef(e.dataId)}removeDataId(e,t){this.state.tensorInfo.has(e)&&this.state.tensorInfo.get(e).backend===t&&(this.state.tensorInfo.delete(e),this.state.numDataBuffers--)}disposeTensor(e){if(!this.state.tensorInfo.has(e.dataId))return;let t=this.state.tensorInfo.get(e.dataId);if(this.state.numTensors--,e.dtype===`string`&&(this.state.numStringTensors--,this.state.numBytes-=t.bytes),e.dtype!==`complex64`&&e.dtype!==`string`){let t=e.size*A(e.dtype);this.state.numBytes-=t}t.backend.disposeData(e.dataId)&&this.removeDataId(e.dataId,t.backend)}disposeVariables(){for(let e in this.state.registeredVariables){let t=this.state.registeredVariables[e];this.disposeVariable(t)}}disposeVariable(e){this.disposeTensor(e),this.state.registeredVariables[e.name]!=null&&delete this.state.registeredVariables[e.name]}memory(){let e=this.backend.memory();return e.numTensors=this.state.numTensors,e.numDataBuffers=this.state.numDataBuffers,e.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(e.unreliable=!0,e.reasons??=[],e.reasons.push(`Memory usage by string tensors is approximate (2 bytes per character)`)),e}async profile(e){this.state.profiling=!0;let t=this.state.numBytes,n=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await e(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(e=>e.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-t,this.state.activeProfile.newTensors=this.state.numTensors-n;for(let e of this.state.activeProfile.kernels)e.kernelTimeMs=await e.kernelTimeMs,e.extraInfo=await e.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(e,t,n,r,i,a){let o={id:this.state.nextTapeNodeId++,kernelName:e,inputs:t,outputs:n,saved:i},s=yr(e);s!=null&&(r=s.gradFunc),r!=null&&(o.gradient=e=>(e=e.map((e,t)=>{if(e==null){let e=n[t],r=M(e.size,e.dtype);return this.makeTensor(r,e.shape,e.dtype)}return e}),r(e.length>1?e:e[0],i,a))),this.state.activeTape.push(o)}keep(e){return e.kept=!0,e}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(e){let t={track:[],name:`unnamed scope`,id:this.state.nextScopeId++};e&&(t.name=e),this.state.scopeStack.push(t),this.state.activeScope=t}endScope(e){let t=Ti(e),n=new Set(t.map(e=>e.id));for(let e=0;e<this.state.activeScope.track.length;e++){let t=this.state.activeScope.track[e];!t.kept&&!n.has(t.id)&&t.dispose()}let r=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],t.forEach(e=>{!e.kept&&e.scopeId===r.id&&this.track(e)})}gradients(e,t,n,r=!1){if(f(t.length>0,()=>`gradients() received an empty list of xs.`),n!=null&&n.dtype!==`float32`)throw Error(`dy must have 'float32' dtype, but has '${n.dtype}'`);let i=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy(`forward`,e));f(i instanceof fi,()=>`The result y returned by f() must be a tensor.`);let a=Xr(this.state.activeTape,t,i);if(!r&&a.length===0&&t.length>0)throw Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.`);return this.tidy(`backward`,()=>{let e={};e[i.id]=n??ji(i.shape),Zr(e,a,e=>this.tidy(e),Ni);let r=t.map(t=>e[t.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(e=>{for(let t of e.saved)t.dispose()}),this.state.activeTape=null),{value:i,grads:r}})}customGrad(e){return f(oe(e),()=>`The f passed in customGrad(f) must be a function.`),(...t)=>{f(t.every(e=>e instanceof fi),()=>`The args passed in customGrad(f)(x1, x2,...) must all be tensors`);let n,r={};return t.forEach((e,t)=>{r[t]=e}),this.runKernelFunc({forwardFunc:(r,i)=>(n=e(...t,i),f(n.value instanceof fi,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),f(oe(n.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),n.value),backwardsFunc:(e,r)=>{let i=n.gradFunc(e,r),a=Array.isArray(i)?i:[i];f(a.length===t.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),f(a.every(e=>e instanceof fi),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");let o={};return a.forEach((e,t)=>{o[t]=()=>e}),o},inputs:r})}}readSync(e){return this.state.tensorInfo.get(e).backend.readSync(e)}read(e){return this.state.tensorInfo.get(e).backend.read(e)}readToGPU(e,t){return this.state.tensorInfo.get(e).backend.readToGPU(e,t)}async time(e){let t=Wr(),n=await this.backend.time(e);return n.wallMs=Wr()-t,n}track(e){return this.state.activeScope!=null&&(e.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(e)),e}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new ki;for(let e in this.registry)this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}};Ai.nextTensorId=0,Ai.nextVariableId=0;function ji(e){let t=ue(g(e),`float32`);return P.makeTensor(t,e,`float32`)}function Mi(){let e=we();return e._tfengine??=new Ai(new _e(e)),Se(e._tfengine.ENV),ui(()=>e._tfengine),e._tfengine}var P=Mi();function Ni(e,t){let n={a:e,b:t};return P.runKernel(`Add`,n)}function Pi(){return typeof navigator<`u`&&navigator!=null}function Fi(e){if(e||Pi()){if(e||=navigator,e.product===`ReactNative`)return!0;let t=e.userAgent||e.vendor||(typeof window<`u`?window.opera:``);if(!t){let t=e;return t.userAgentData&&t.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function Ii(){return typeof window<`u`&&window.document!=null||typeof WorkerGlobalScope<`u`}var Li=N();Li.registerFlag(`DEBUG`,()=>!1,e=>{e&&console.warn(`Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.`)}),Li.registerFlag(`IS_BROWSER`,()=>Ii()),Li.registerFlag(`IS_NODE`,()=>typeof process<`u`&&process.versions!==void 0&&process.versions.node!==void 0),Li.registerFlag(`IS_CHROME`,()=>typeof navigator<`u`&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor)),Li.registerFlag(`PROD`,()=>!1),Li.registerFlag(`TENSORLIKE_CHECK_SHAPE_CONSISTENCY`,()=>Li.getBool(`DEBUG`)),Li.registerFlag(`DEPRECATION_WARNINGS_ENABLED`,()=>!0),Li.registerFlag(`IS_TEST`,()=>!1),Li.registerFlag(`CHECK_COMPUTATION_FOR_ERRORS`,()=>!0),Li.registerFlag(`WRAP_TO_IMAGEBITMAP`,()=>!1),Li.registerFlag(`ENGINE_COMPILE_ONLY`,()=>!1),Li.registerFlag(`CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU`,()=>!1),Li.registerFlag(`USE_SETTIMEOUTCUSTOM`,()=>!1);function Ri(e,t){let n=e;if(k(e))return t===`string`?[]:[e.length];if(!Array.isArray(e))return[];let r=[];for(;Array.isArray(n)||k(n)&&t!==`string`;)r.push(n.length),n=n[0];return Array.isArray(e)&&N().getBool(`TENSORLIKE_CHECK_SHAPE_CONSISTENCY`)&&zi(e,r,[]),r}function zi(e,t,n){if(n||=[],!Array.isArray(e)&&!k(e)){f(t.length===0,()=>`Element arr[${n.join(`][`)}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}f(t.length>0,()=>`Element arr[${n.join(`][`)}] should be a primitive, but is an array of ${e.length} elements`),f(e.length===t[0],()=>`Element arr[${n.join(`][`)}] should have ${t[0]} elements, but has ${e.length} elements`);let r=t.slice(1);for(let t=0;t<e.length;++t)zi(e[t],r,n.concat(t))}function Bi(e,t,n,r){if(e!==`string_or_numeric`){if(e==null)throw Error(`Expected dtype cannot be null.`);if(e!==`numeric`&&e!==t||e===`numeric`&&t===`string`)throw Error(`Argument '${n}' passed to '${r}' must be ${e} tensor, but got ${t} tensor`)}}function F(e,t,n,r=`numeric`){if(e instanceof fi)return Bi(r,e.dtype,t,n),e;let i=ae(e);if(i!==`string`&&[`bool`,`int32`,`float32`].indexOf(r)>=0&&(i=r),Bi(r,i,t,n),e==null||!k(e)&&!Array.isArray(e)&&typeof e!=`number`&&typeof e!=`boolean`&&typeof e!=`string`){let r=e==null?`null`:e.constructor.name;throw Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${r}'`)}let a=Ri(e,i);!k(e)&&!Array.isArray(e)&&(e=[e]);let o=i===`string`?h(e,[],!0):Ur(e,i);return P.makeTensor(o,a,i)}function Vi(e,t,n,r=`numeric`){if(!Array.isArray(e))throw Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((e,i)=>F(e,`${t}[${i}]`,n,r))}var Hi=`__op`;function I(e){let t=Object.keys(e);if(t.length!==1)throw Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0],r=e[n];n.endsWith(`_`)&&(n=n.substring(0,n.length-1)),n+=Hi;let i=(...e)=>{P.startScope(n);try{let t=r(...e);return he(t)&&console.error(`Cannot return a Promise inside of tidy.`),P.endScope(t),t}catch(e){throw P.endScope(null),e}};return Object.defineProperty(i,"name",{value:n,configurable:!0}),i}function Ui(e,t){let n=F(e,`real`,`complex`),r=F(t,`imag`,`complex`);p(n.shape,r.shape,`real and imag shapes, ${n.shape} and ${r.shape}, must match in call to tf.complex().`);let i={real:n,imag:r};return P.runKernel(Je,i)}var Wi=I({complex_:Ui});function Gi(e,t,n,r){if(r??=ae(e),r===`complex64`)throw Error(`Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).`);if(!k(e)&&!Array.isArray(e)&&typeof e!=`number`&&typeof e!=`boolean`&&typeof e!=`string`)throw Error(`values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray`);if(t!=null){fe(t);let e=g(t),r=g(n);f(e===r,()=>`Based on the provided shape, [${t}], the tensor should have ${e} values but has ${r}`);for(let e=0;e<n.length;++e){let r=n[e],i=e!==n.length-1||r!==g(t.slice(e));f(n[e]===t[e]||!i,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return!k(e)&&!Array.isArray(e)&&(e=[e]),t||=n,e=r===`string`?h(e,[],!0):Ur(e,r),P.makeTensor(e,t,r)}function Ki(e,t,n){return Gi(e,t,Ri(e,n),n)}var qi={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8},Ji=4;async function Yi(e,t){let n=[],r=[],i=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);for(let a=0;a<i.length;++a){let o=i[a],s=Array.isArray(e)?e[a].tensor:e[o];if(s.dtype!==`float32`&&s.dtype!==`int32`&&s.dtype!==`bool`&&s.dtype!==`string`&&s.dtype!==`complex64`)throw Error(`Unsupported dtype in weight '${o}': ${s.dtype}`);let c={name:o,shape:s.shape,dtype:s.dtype};if(s.dtype===`string`){let e=new Promise(async e=>{let t=await s.bytes(),n=t.reduce((e,t)=>e+t.length,0)+Ji*t.length,r=new Uint8Array(n),i=0;for(let e=0;e<t.length;e++){let n=t[e],a=new Uint8Array(new Uint32Array([n.length]).buffer);r.set(a,i),i+=Ji,r.set(n,i),i+=n.length}e(r)});r.push(e)}else r.push(s.data());t!=null&&(c.group=t),n.push(c)}return{data:Zi(await Promise.all(r)),specs:n}}function Xi(e,t){let n={},r,i=0;for(let a of t){let t=a.name,o=a.dtype,s=a.shape,c=g(s),l;if(`quantization`in a){let n=a.quantization;if(n.dtype===`uint8`||n.dtype===`uint16`){if(!(`min`in n&&`scale`in n))throw Error(`Weight ${a.name} with quantization ${n.dtype} doesn't have corresponding metadata min and scale.`)}else if(n.dtype===`float16`){if(o!==`float32`)throw Error(`Weight ${a.name} is quantized with ${n.dtype} which only supports weights of type float32 not ${o}.`)}else throw Error(`Weight ${a.name} has unknown quantization dtype ${n.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);let s=qi[n.dtype],u=e.slice(i,i+c*s),d=n.dtype===`uint8`?new Uint8Array(u):new Uint16Array(u);if(o===`float32`){if(n.dtype===`uint8`||n.dtype===`uint16`){l=new Float32Array(d.length);for(let e=0;e<d.length;e++){let t=d[e];l[e]=t*n.scale+n.min}}else if(n.dtype===`float16`)r===void 0&&(r=fa()),l=r(d);else throw Error(`Unsupported quantization type ${n.dtype} for weight type float32.`)}else if(o===`int32`){if(n.dtype!==`uint8`&&n.dtype!==`uint16`)throw Error(`Unsupported quantization type ${n.dtype} for weight type int32.`);l=new Int32Array(d.length);for(let e=0;e<d.length;e++){let t=d[e];l[e]=Math.round(t*n.scale+n.min)}}else throw Error(`Unsupported dtype in weight '${t}': ${o}`);i+=c*s}else if(o===`string`){let t=g(a.shape);l=[];for(let n=0;n<t;n++){let t=new Uint32Array(e.slice(i,i+Ji))[0];i+=Ji;let n=new Uint8Array(e.slice(i,i+t));l.push(n),i+=t}}else{let r=qi[o],a=e.slice(i,i+c*r);if(o===`float32`)l=new Float32Array(a);else if(o===`int32`)l=new Int32Array(a);else if(o===`bool`)l=new Uint8Array(a);else if(o===`complex64`){l=new Float32Array(a);let e=new Float32Array(l.length/2),r=new Float32Array(l.length/2);for(let t=0;t<e.length;t++)e[t]=l[t*2],r[t]=l[t*2+1];let i=Ki(e,s,`float32`),o=Ki(r,s,`float32`);n[t]=Wi(i,o),i.dispose(),o.dispose()}else throw Error(`Unsupported dtype in weight '${t}': ${o}`);i+=c*r}o!==`complex64`&&(n[t]=Ki(l,s,o))}return n}function Zi(e){if(e===null)throw Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0,n=[];e.forEach(e=>{if(t+=e.byteLength,n.push(e.byteLength===e.buffer.byteLength?e:new e.constructor(e)),!(e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array))throw Error(`Unsupported TypedArray subtype: ${e.constructor.name}`)});let r=new Uint8Array(t),i=0;return n.forEach(e=>{r.set(new Uint8Array(e.buffer),i),i+=e.byteLength}),r.buffer}var Qi=typeof Buffer<`u`&&(typeof Blob>`u`||typeof atob>`u`||typeof btoa>`u`);function $i(e){return Qi?Buffer.byteLength(e):new Blob([e]).size}function ea(e){if(Qi)return Buffer.from(e).toString(`base64`);let t=new Uint8Array(e),n=``;for(let e=0,r=t.length;e<r;e++)n+=String.fromCharCode(t[e]);return btoa(n)}function ta(e){if(Qi){let t=Buffer.from(e,`base64`);return t.buffer.slice(t.byteOffset,t.byteOffset+t.byteLength)}let t=atob(e),n=new Uint8Array(t.length);for(let e=0;e<t.length;++e)n.set([t.charCodeAt(e)],e);return n.buffer}function na(e){if(e.length===1)return e[0];let t=0;e.forEach(e=>{t+=e.byteLength});let n=new Uint8Array(t),r=0;return e.forEach(e=>{n.set(new Uint8Array(e),r),r+=e.byteLength}),n.buffer}function ra(e){for(e=e.trim();e.endsWith(`/`);)e=e.slice(0,e.length-1);let t=e.split(`/`);return t[t.length-1]}function ia(e,t){let n={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,weightsManifest:t};return e.signature!=null&&(n.signature=e.signature),e.userDefinedMetadata!=null&&(n.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(n.modelInitializer=e.modelInitializer),e.trainingConfig!=null&&(n.trainingConfig=e.trainingConfig),n}function aa(e,t,n){let r={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy};if(e.trainingConfig!=null&&(r.trainingConfig=e.trainingConfig),e.weightsManifest!=null){if(!t)throw Error(`modelJSON has weightsManifest but weightSpecs is null`);if(!n)throw Error(`modelJSON has weightsManifest but weightData is null`);r.weightSpecs=t,r.weightData=n}return e.signature!=null&&(r.signature=e.signature),e.userDefinedMetadata!=null&&(r.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(r.modelInitializer=e.modelInitializer),r}async function oa(e,t){let n,r;return e.weightsManifest!=null&&([n,r]=await t(e.weightsManifest)),aa(e,n,r)}function sa(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`Expected JSON model topology, received ArrayBuffer.`);return{dateSaved:new Date,modelTopologyType:`JSON`,modelTopologyBytes:e.modelTopology==null?0:$i(JSON.stringify(e.modelTopology)),weightSpecsBytes:e.weightSpecs==null?0:$i(JSON.stringify(e.weightSpecs)),weightDataBytes:e.weightData==null?0:e.weightData.byteLength}}function ca(e){let t=[];for(let n of e)t.push(...n.weights);return t}function la(){let e=e=>{let t=e<<13,n=0;for(;!(t&8388608);)n-=8388608,t<<=1;return t&=-8388609,n+=947912704,t|n},t=new Uint32Array(2048);t[0]=0;for(let n=1;n<1024;n++)t[n]=e(n);for(let e=1024;e<2048;e++)t[e]=939524096+(e-1024<<13);return t}function ua(){let e=new Uint32Array(64);e[0]=0,e[31]=1199570944,e[32]=2147483648,e[63]=3347054592;for(let t=1;t<31;t++)e[t]=t<<23;for(let t=33;t<63;t++)e[t]=2147483648+(t-32<<23);return e}function da(){let e=new Uint32Array(64);for(let t=0;t<64;t++)e[t]=1024;return e[0]=e[32]=0,e}function fa(){let e=la(),t=ua(),n=da();return r=>{let i=new ArrayBuffer(4*r.length),a=new Uint32Array(i);for(let i=0;i<r.length;i++){let o=r[i],s=e[n[o>>10]+(o&1023)]+t[o>>10];a[i]=s}return new Float32Array(i)}}var pa=class e{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return e.instance??=new e,e.instance}static registerSaveRouter(t){e.getInstance().saveRouters.push(t)}static registerLoadRouter(t){e.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return e.getHandlers(t,`save`)}static getLoadHandlers(t,n){return e.getHandlers(t,`load`,n)}static getHandlers(t,n,r){let i=[];return(n===`load`?e.getInstance().loadRouters:e.getInstance().saveRouters).forEach(e=>{let n=e(t,r);n!==null&&i.push(n)}),i}},ma=e=>pa.registerSaveRouter(e),ha=e=>pa.registerLoadRouter(e),ga=e=>pa.getSaveHandlers(e),_a=(e,t)=>pa.getLoadHandlers(e,t),va=`tensorflowjs`,ya=1,ba=`models_store`,xa=`model_info_store`;function Sa(){if(!N().getBool(`IS_BROWSER`))throw Error(`Failed to obtain IndexedDB factory because the current environmentis not a web browser.`);let e=typeof window>`u`?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(t==null)throw Error(`The current browser does not appear to support IndexedDB.`);return t}function Ca(e){let t=e.result;t.createObjectStore(ba,{keyPath:`modelPath`}),t.createObjectStore(xa,{keyPath:`modelPath`})}var wa=class{constructor(e){if(this.indexedDB=Sa(),e==null||!e)throw Error(`For IndexedDB, modelPath must not be null, undefined or empty.`);this.modelPath=e}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserLocalStorage.save() does not support saving model topology in binary formats yet.`);return this.databaseAction(this.modelPath,e)}async load(){return this.databaseAction(this.modelPath)}databaseAction(e,t){return new Promise((e,n)=>{let r=this.indexedDB.open(va,ya);r.onupgradeneeded=()=>Ca(r),r.onsuccess=()=>{let i=r.result;if(t==null){let t=i.transaction(ba,`readonly`),r=t.objectStore(ba).get(this.modelPath);r.onsuccess=()=>{if(r.result==null)return i.close(),n(Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));e(r.result.modelArtifacts)},r.onerror=e=>(i.close(),n(r.error)),t.oncomplete=()=>i.close()}else{let r=sa(t),a=i.transaction(xa,`readwrite`),o=a.objectStore(xa),s=o.put({modelPath:this.modelPath,modelArtifactsInfo:r}),c;s.onsuccess=()=>{c=i.transaction(ba,`readwrite`);let s=c.objectStore(ba).put({modelPath:this.modelPath,modelArtifacts:t,modelArtifactsInfo:r});s.onsuccess=()=>e({modelArtifactsInfo:r}),s.onerror=e=>{o=a.objectStore(xa);let t=o.delete(this.modelPath);t.onsuccess=()=>(i.close(),n(s.error)),t.onerror=e=>(i.close(),n(s.error))}},s.onerror=e=>(i.close(),n(s.error)),a.oncomplete=()=>{c==null?i.close():c.oncomplete=()=>i.close()}}},r.onerror=e=>n(r.error)})}};wa.URL_SCHEME=`indexeddb://`;var Ta=e=>N().getBool(`IS_BROWSER`)&&!Array.isArray(e)&&e.startsWith(wa.URL_SCHEME)?Ea(e.slice(wa.URL_SCHEME.length)):null;pa.registerSaveRouter(Ta),pa.registerLoadRouter(Ta);function Ea(e){return new wa(e)}function Da(e){return e.startsWith(wa.URL_SCHEME)?e.slice(wa.URL_SCHEME.length):e}var Oa=class{constructor(){this.indexedDB=Sa()}async listModels(){return new Promise((e,t)=>{let n=this.indexedDB.open(va,ya);n.onupgradeneeded=()=>Ca(n),n.onsuccess=()=>{let r=n.result,i=r.transaction(xa,`readonly`),a=i.objectStore(xa).getAll();a.onsuccess=()=>{let t={};for(let e of a.result)t[e.modelPath]=e.modelArtifactsInfo;e(t)},a.onerror=e=>(r.close(),t(a.error)),i.oncomplete=()=>r.close()},n.onerror=e=>t(n.error)})}async removeModel(e){return e=Da(e),new Promise((t,n)=>{let r=this.indexedDB.open(va,ya);r.onupgradeneeded=()=>Ca(r),r.onsuccess=()=>{let i=r.result,a=i.transaction(xa,`readwrite`),o=a.objectStore(xa),s=o.get(e),c;s.onsuccess=()=>{if(s.result==null)return i.close(),n(Error(`Cannot find model with path '${e}' in IndexedDB.`));{let r=o.delete(e),a=()=>{c=i.transaction(ba,`readwrite`);let r=c.objectStore(ba).delete(e);r.onsuccess=()=>t(s.result.modelArtifactsInfo),r.onerror=e=>n(s.error)};r.onsuccess=a,r.onerror=e=>(a(),i.close(),n(s.error))}},s.onerror=e=>(i.close(),n(s.error)),a.oncomplete=()=>{c==null?i.close():c.oncomplete=()=>i.close()}},r.onerror=e=>n(r.error)})}},ka=`/`,Aa=`tensorflowjs_models`,ja=`info`,Ma=`model_topology`,Na=`weight_specs`,Pa=`weight_data`,Fa=`model_metadata`;function Ia(e){return{info:[Aa,e,ja].join(ka),topology:[Aa,e,Ma].join(ka),weightSpecs:[Aa,e,Na].join(ka),weightData:[Aa,e,Pa].join(ka),modelMetadata:[Aa,e,Fa].join(ka)}}function La(e){for(let t of Object.values(e))window.localStorage.removeItem(t)}function Ra(e){let t=e.split(ka);if(t.length<3)throw Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(ka)}function za(e){return e.startsWith(Ba.URL_SCHEME)?e.slice(Ba.URL_SCHEME.length):e}var Ba=class{constructor(e){if(!N().getBool(`IS_BROWSER`)||typeof window>`u`||window.localStorage===void 0)throw Error(`The current environment does not support local storage.`);if(this.LS=window.localStorage,e==null||!e)throw Error(`For local storage, modelPath must not be null, undefined or empty.`);this.modelPath=e,this.keys=Ia(this.modelPath)}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserLocalStorage.save() does not support saving model topology in binary formats yet.`);{let t=JSON.stringify(e.modelTopology),n=JSON.stringify(e.weightSpecs),r=sa(e);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,t),this.LS.setItem(this.keys.weightSpecs,n),this.LS.setItem(this.keys.weightData,ea(e.weightData));let i={format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,signature:e.signature==null?void 0:e.signature,userDefinedMetadata:e.userDefinedMetadata==null?void 0:e.userDefinedMetadata,modelInitializer:e.modelInitializer==null?void 0:e.modelInitializer,trainingConfig:e.trainingConfig==null?void 0:e.trainingConfig};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(i)),{modelArtifactsInfo:r}}catch{throw La(this.keys),Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){let e=JSON.parse(this.LS.getItem(this.keys.info));if(e==null)throw Error(`In local storage, there is no model with name '${this.modelPath}'`);if(e.modelTopologyType!==`JSON`)throw Error(`BrowserLocalStorage does not support loading non-JSON model topology yet.`);let t={},n=JSON.parse(this.LS.getItem(this.keys.topology));if(n==null)throw Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);t.modelTopology=n;let r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(r==null)throw Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);t.weightSpecs=r;let i=this.LS.getItem(this.keys.modelMetadata);if(i!=null){let e=JSON.parse(i);t.format=e.format,t.generatedBy=e.generatedBy,t.convertedBy=e.convertedBy,e.signature!=null&&(t.signature=e.signature),e.userDefinedMetadata!=null&&(t.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(t.modelInitializer=e.modelInitializer),e.trainingConfig!=null&&(t.trainingConfig=e.trainingConfig)}let a=this.LS.getItem(this.keys.weightData);if(a==null)throw Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return t.weightData=ta(a),t}};Ba.URL_SCHEME=`localstorage://`;var Va=e=>N().getBool(`IS_BROWSER`)&&!Array.isArray(e)&&e.startsWith(Ba.URL_SCHEME)?Ha(e.slice(Ba.URL_SCHEME.length)):null;pa.registerSaveRouter(Va),pa.registerLoadRouter(Va);function Ha(e){return new Ba(e)}var Ua=class{constructor(){f(N().getBool(`IS_BROWSER`),()=>`Current environment is not a web browser`),f(typeof window>`u`||window.localStorage!==void 0,()=>`Current browser does not appear to support localStorage`),this.LS=window.localStorage}async listModels(){let e={};for(let t=0;t<this.LS.length;++t){let n=this.LS.key(t);if(n.startsWith(`tensorflowjs_models/`)&&n.endsWith(`/info`)){let t=Ra(n);e[t]=JSON.parse(this.LS.getItem(n))}}return e}async removeModel(e){e=za(e);let t=Ia(e);if(this.LS.getItem(t.info)==null)throw Error(`Cannot find model at path '${e}'`);let n=JSON.parse(this.LS.getItem(t.info));return La(t),n}},Wa=`://`,Ga=class e{constructor(){this.managers={}}static getInstance(){return e.instance??=new e,e.instance}static registerManager(t,n){f(t!=null,()=>`scheme must not be undefined or null.`),t.endsWith(Wa)&&(t=t.slice(0,t.indexOf(Wa))),f(t.length>0,()=>`scheme must not be an empty string.`);let r=e.getInstance();f(r.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),r.managers[t]=n}static getManager(t){let n=e.getInstance().managers[t];if(n==null)throw Error(`Cannot find model manager for scheme '${t}'`);return n}static getSchemes(){return Object.keys(e.getInstance().managers)}};function Ka(e){if(e.indexOf(Wa)===-1)throw Error(`The url string provided does not contain a scheme. Supported schemes are: ${Ga.getSchemes().join(`,`)}`);return{scheme:e.split(Wa)[0],path:e.split(Wa)[1]}}async function qa(e,t,n=!1){f(e!==t,()=>`Old path and new path are the same: '${e}'`);let r=pa.getLoadHandlers(e);f(r.length>0,()=>`Copying failed because no load handler is found for source URL ${e}.`),f(r.length<2,()=>`Copying failed because more than one (${r.length}) load handlers for source URL ${e}.`);let i=r[0],a=pa.getSaveHandlers(t);f(a.length>0,()=>`Copying failed because no save handler is found for destination URL ${t}.`),f(a.length<2,()=>`Copying failed because more than one (${r.length}) save handlers for destination URL ${t}.`);let o=a[0],s=Ka(e).scheme,c=Ka(e).path,l=s===Ka(e).scheme,u=await i.load();n&&l&&await Ga.getManager(s).removeModel(c);let d=await o.save(u);return n&&!l&&await Ga.getManager(s).removeModel(c),d.modelArtifactsInfo}async function Ja(){let e=Ga.getSchemes(),t={};for(let n of e){let e=await Ga.getManager(n).listModels();for(let r in e){let i=n+Wa+r;t[i]=e[r]}}return t}async function Ya(e){let t=Ka(e);return Ga.getManager(t.scheme).removeModel(t.path)}async function Xa(e,t){return qa(e,t,!1)}async function Za(e,t){return qa(e,t,!0)}var Qa=class{constructor(){this.messageName=`setTimeoutCustom`,this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(e,t){return fetch(e,t)}now(){return performance.now()}encode(e,t){if(t!==`utf-8`&&t!==`utf8`)throw Error(`Browser's encoder only supports utf-8, but got ${t}`);return this.textEncoder??=new TextEncoder,this.textEncoder.encode(e)}decode(e,t){return new TextDecoder(t).decode(e)}setTimeoutCustom(e,t){!window||!N().getBool(`USE_SETTIMEOUTCUSTOM`)?setTimeout(e,t):(this.functionRefs.push(e),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},`*`)},t),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener(`message`,e=>{if(e.source===window&&e.data.name===this.messageName){e.stopPropagation();let t=this.functionRefs[e.data.index];t(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0)))}};if(N().get(`IS_BROWSER`)){N().setPlatform(`browser`,new Qa);try{Ga.registerManager(Ba.URL_SCHEME,new Ua)}catch{}try{Ga.registerManager(wa.URL_SCHEME,new Oa)}catch{}}var $a=e(((e,t)=>{t.exports={}})),eo={importFetch:()=>$a()},to,no=class{constructor(){this.util=$a(),this.textEncoder=new this.util.TextEncoder}fetch(e,t){return N().global.fetch==null?(to??=eo.importFetch(),to(e,t)):N().global.fetch(e,t)}now(){let e=process.hrtime();return e[0]*1e3+e[1]/1e6}encode(e,t){if(t!==`utf-8`&&t!==`utf8`)throw Error(`Node built-in encoder only supports utf-8, but got ${t}`);return this.textEncoder.encode(e)}decode(e,t){return e.length===0?``:new this.util.TextDecoder(t).decode(e)}};N().get(`IS_NODE`)&&!N().get(`IS_BROWSER`)&&N().setPlatform(`node`,new no);function L(e,t=`float32`,n){return t||=`float32`,fe(e),new si(e,t,n)}function ro(e,t){let n=F(e,`x`,`cast`);if(!O(t))throw Error(`Failed to cast to unknown dtype ${t}`);if(t===`string`&&n.dtype!==`string`||t!==`string`&&n.dtype===`string`)throw Error(`Only strings can be casted to strings`);let r={x:n},i={dtype:t};return P.runKernel(Ge,r,i)}var io=I({cast_:ro});function ao(e){let t={x:F(e,`x`,`clone`,`string_or_numeric`)};return P.runKernel(jt,t)}var oo=I({clone_:ao});function so(e,t=!1){console.log(e.toString(t))}Mi(),di({buffer:L,cast:io,clone:oo,print:so});var co=`model`,lo=`.json`,uo=`.weights.bin`;function fo(e){return new Promise(e=>setTimeout(e)).then(e)}var po=class e{constructor(t){if(!N().getBool(`IS_BROWSER`))throw Error(`browserDownloads() cannot proceed because the current environment is not a browser.`);t.startsWith(e.URL_SCHEME)&&(t=t.slice(e.URL_SCHEME.length)),(t==null||t.length===0)&&(t=co),this.modelJsonFileName=t+lo,this.weightDataFileName=t+uo}async save(e){if(typeof document>`u`)throw Error("Browser downloads are not supported in this environment since `document` is not present");let t=window.URL.createObjectURL(new Blob([e.weightData],{type:`application/octet-stream`}));if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserDownloads.save() does not support saving model topology in binary formats yet.`);{let n=ia(e,[{paths:[`./`+this.weightDataFileName],weights:e.weightSpecs}]),r=window.URL.createObjectURL(new Blob([JSON.stringify(n)],{type:`application/json`})),i=this.modelJsonAnchor==null?document.createElement(`a`):this.modelJsonAnchor;if(i.download=this.modelJsonFileName,i.href=r,await fo(()=>i.dispatchEvent(new MouseEvent(`click`))),e.weightData!=null){let e=this.weightDataAnchor==null?document.createElement(`a`):this.weightDataAnchor;e.download=this.weightDataFileName,e.href=t,await fo(()=>e.dispatchEvent(new MouseEvent(`click`)))}return{modelArtifactsInfo:sa(e)}}}};po.URL_SCHEME=`downloads://`;var mo=class{constructor(e){if(e==null||e.length<1)throw Error(`When calling browserFiles, at least 1 file is required, but received ${e}`);this.jsonFile=e[0],this.weightsFiles=e.slice(1)}async load(){return new Promise((e,t)=>{let n=new FileReader;n.onload=n=>{let r=JSON.parse(n.target.result),i=r.modelTopology;i==null?t(Error(`modelTopology field is missing from file ${this.jsonFile.name}`)):r.weightsManifest==null?t(Error(`weightManifest field is missing from file ${this.jsonFile.name}`)):this.weightsFiles.length===0?e({modelTopology:i}):e(oa(r,e=>this.loadWeights(e)))},n.onerror=e=>t(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),n.readAsText(this.jsonFile)})}loadWeights(e){let t=[],n=[];for(let r of e)t.push(...r.weights),n.push(...r.paths);let r=this.checkManifestAndWeightFiles(e),i=n.map(e=>this.loadWeightsFile(e,r[e]));return Promise.all(i).then(e=>[t,na(e)])}loadWeightsFile(e,t){return new Promise((n,r)=>{let i=new FileReader;i.onload=e=>{let t=e.target.result;n(t)},i.onerror=t=>r(`Failed to weights data from file of path '${e}'.`),i.readAsArrayBuffer(t)})}checkManifestAndWeightFiles(e){let t=[],n=this.weightsFiles.map(e=>ra(e.name)),r={};for(let i of e)i.paths.forEach(e=>{let i=ra(e);if(t.indexOf(i)!==-1)throw Error(`Duplicate file basename found in weights manifest: '${i}'`);if(t.push(i),n.indexOf(i)===-1)throw Error(`Weight file with basename '${i}' is not provided.`);r[e]=this.weightsFiles[n.indexOf(i)]});if(t.length!==this.weightsFiles.length)throw Error(`Mismatch in the number of files in weights manifest (${t.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}};pa.registerSaveRouter(e=>N().getBool(`IS_BROWSER`)&&!Array.isArray(e)&&e.startsWith(po.URL_SCHEME)?ho(e.slice(po.URL_SCHEME.length)):null);function ho(e=`model`){return new po(e)}function go(e){return new mo(e)}function _o(e,t,n,r){o(e),n??=0,r??=1,s(n,r);let i=0,a=a=>(a.then(a=>(t(n+ ++i/e.length*(r-n)),a)),a);function o(e){f(e!=null&&Array.isArray(e)&&e.length>0,()=>`promises must be a none empty array`)}function s(e,t){f(e>=0&&e<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${e}`),f(t>=0&&t<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${t}`),f(t>=e,()=>`startFraction must be no more than endFraction, but got startFraction ${e} and endFraction ${t}`)}return Promise.all(e.map(a))}async function vo(e,t){t??={};let n=t.fetchFunc==null?N().platform.fetch:t.fetchFunc,r=e.map(e=>n(e,t.requestInit,{isBinary:!0})),i=(t.onProgress==null?await Promise.all(r):await _o(r,t.onProgress,0,.5)).map(e=>e.arrayBuffer());return t.onProgress==null?await Promise.all(i):await _o(i,t.onProgress,.5,1)}async function yo(e,t=``,n,r){return bo(e=>vo(e,{requestInit:r}))(e,t,n)}function bo(e){return async(t,n=``,r)=>{let i=t.map(()=>!1),a={},o=r==null?[]:r.map(()=>!1),s=[];if(t.forEach((e,t)=>{let n=0;e.weights.forEach(e=>{let c=qi[`quantization`in e?e.quantization.dtype:e.dtype]*g(e.shape),l=()=>{i[t]=!0,a[t]??(a[t]=[]),a[t].push({manifestEntry:e,groupOffset:n,sizeBytes:c})};r==null?l():r.forEach((t,n)=>{t===e.name&&(l(),o[n]=!0)}),s.push(e.name),n+=c})}),!o.every(e=>e)){let e=r.filter((e,t)=>!o[t]);throw Error(`Could not find weights in manifest with names: ${e.join(`, `)}. \nManifest JSON has weights with names: ${s.join(`, `)}.`)}let c=i.reduce((e,t,n)=>(t&&e.push(n),e),[]),l=[];c.forEach(e=>{t[e].paths.forEach(e=>{let t=n+(n.endsWith(`/`)?``:`/`)+e;l.push(t)})});let u=await e(l),d={},f=0;return c.forEach(e=>{let n=t[e].paths.length,r=0;for(let e=0;e<n;e++)r+=u[f+e].byteLength;let i=new ArrayBuffer(r),o=new Uint8Array(i),s=0;for(let e=0;e<n;e++){let t=new Uint8Array(u[f+e]);o.set(t,s),s+=t.byteLength}a[e].forEach(e=>{let t=Xi(i.slice(e.groupOffset,e.groupOffset+e.sizeBytes),[e.manifestEntry]);for(let e in t)d[e]=t[e]}),f+=n}),d}}var xo=`application/octet-stream`,So=`application/json`,Co=class{constructor(e,t){if(this.DEFAULT_METHOD=`POST`,t??={},this.weightPathPrefix=t.weightPathPrefix,this.onProgress=t.onProgress,this.weightUrlConverter=t.weightUrlConverter,t.fetchFunc==null?this.fetch=N().platform.fetch:(f(typeof t.fetchFunc==`function`,()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=t.fetchFunc),f(e!=null&&e.length>0,()=>`URL path for http must not be null, undefined or empty.`),Array.isArray(e)&&f(e.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${e.length}).`),this.path=e,t.requestInit!=null&&t.requestInit.body!=null)throw Error(`requestInit is expected to have no pre-existing body, but has one.`);this.requestInit=t.requestInit||{}}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.`);let t=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);t.body=new FormData;let n=ia(e,[{paths:[`./model.weights.bin`],weights:e.weightSpecs}]);t.body.append(`model.json`,new Blob([JSON.stringify(n)],{type:So}),`model.json`),e.weightData!=null&&t.body.append(`model.weights.bin`,new Blob([e.weightData],{type:xo}),`model.weights.bin`);let r=await this.fetch(this.path,t);if(r.ok)return{modelArtifactsInfo:sa(e),responses:[r]};throw Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${r.status}.`)}async load(){let e=await this.fetch(this.path,this.requestInit);if(!e.ok)throw Error(`Request to ${this.path} failed with status code ${e.status}. Please verify this URL points to the model JSON of the model to load.`);let t;try{t=await e.json()}catch{let e=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(`.pb`)?e+=` Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.`:e+=` Please make sure the server is serving valid JSON for this request.`,Error(e)}let n=t.modelTopology,r=t.weightsManifest;if(n==null&&r==null)throw Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return oa(t,e=>this.loadWeights(e))}async loadWeights(e){let[t,n]=wo(Array.isArray(this.path)?this.path[1]:this.path),r=this.weightPathPrefix||t,i=ca(e),a=[],o=[];for(let t of e)for(let e of t.paths)this.weightUrlConverter==null?a.push(r+e+n):o.push(this.weightUrlConverter(e));return this.weightUrlConverter&&a.push(...await Promise.all(o)),[i,na(await vo(a,{requestInit:this.requestInit,fetchFunc:this.fetch,onProgress:this.onProgress}))]}};Co.URL_SCHEME_REGEX=/^https?:\/\//;function wo(e){let t=e.lastIndexOf(`/`),n=e.lastIndexOf(`?`),r=e.substring(0,t),i=n>t?e.substring(n):``;return[r+`/`,i]}function To(e){return e.match(Co.URL_SCHEME_REGEX)!=null}var Eo=(e,t)=>{if(typeof fetch>`u`&&(t==null||t.fetchFunc==null))return null;{let n=!0;if(n=Array.isArray(e)?e.every(e=>To(e)):To(e),n)return Do(e,t)}return null};pa.registerSaveRouter(Eo),pa.registerLoadRouter(Eo);function Do(e,t){return new Co(e,t)}function Oo(e,t){return Do(e,t)}var ko=class{constructor(e){this.modelArtifacts=e}load(){return this.modelArtifacts}},Ao=class{constructor(e){this.saveHandler=e}save(e){return this.saveHandler(e)}},jo=class{constructor(e){e.load&&(this.load=()=>Promise.resolve(e.load())),e.save&&(this.save=t=>Promise.resolve(e.save(t)))}};function Mo(e,t,n,r){return new jo(No(...arguments))}function No(e,t,n,r){return arguments.length===1?e.modelTopology!=null||e.weightSpecs!=null?new ko(e):(console.warn(`Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release.`),new ko({modelTopology:e})):(console.warn(`Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release.`),new ko({modelTopology:e,weightSpecs:t,weightData:n,trainingConfig:r}))}function Po(e){return new Ao(e)}function Fo(e){return new Ao(e)}var Io=t({browserFiles:()=>go,browserHTTPRequest:()=>Oo,concatenateArrayBuffers:()=>na,copyModel:()=>Xa,decodeWeights:()=>Xi,encodeWeights:()=>Yi,fromMemory:()=>Mo,fromMemorySync:()=>No,getLoadHandlers:()=>_a,getModelArtifactsForJSON:()=>oa,getModelArtifactsForJSONSync:()=>aa,getModelArtifactsInfoForJSON:()=>sa,getSaveHandlers:()=>ga,getWeightSpecs:()=>ca,http:()=>Do,isHTTPScheme:()=>To,listModels:()=>Ja,loadWeights:()=>yo,moveModel:()=>Za,registerLoadRouter:()=>ha,registerSaveRouter:()=>ma,removeModel:()=>Ya,weightsLoaderFactory:()=>bo,withSaveHandler:()=>Po,withSaveHandlerSync:()=>Fo});function Lo(e,t,n=!1,r=!1){let i=F(e,`a`,`matMul`),a=F(t,`b`,`matMul`);[i,a]=Ci(i,a);let o={a:i,b:a},s={transposeA:n,transposeB:r};return P.runKernel(Ve,o,s)}var R=I({matMul_:Lo});function Ro(e,t,n=1,r=0,i=`int32`){if(t<2)throw Error(`Error in oneHot: depth must be >=2, but it is ${t}`);let a={indices:F(e,`indices`,`oneHot`,`int32`)},o={dtype:i,depth:t,onValue:n,offValue:r};return P.runKernel(ln,a,o)}var zo=I({oneHot_:Ro});function Bo(){return P}function Vo(e,t){return P.tidy(e,t)}function Ho(e){Ti(e).forEach(e=>e.dispose())}function Uo(e){return P.keep(e)}function Wo(e){return P.setBackend(e)}function Go(){return P.ready()}function Ko(){return P.backendName}function qo(e){return P.findBackendFactory(e)}function Jo(e,t,n=1){return P.registerBackend(e,t,n)}function Yo(e){let t={input:F(e,`input`,`imag`)};return P.runKernel(Nt,t)}var Xo=I({imag_:Yo});function Zo(e){let t={x:F(e,`x`,`neg`)};return P.runKernel(`Neg`,t)}var Qo=I({neg_:Zo});function $o(e){let t={input:F(e,`input`,`real`)};return P.runKernel(_n,t)}var es=I({real_:$o});function ts(e,t,n){let r=F(e,`x`,`transpose`);if(t??=r.shape.map((e,t)=>t).reverse(),f(r.rank===t.length,()=>`Error in transpose: rank of input ${r.rank} must match length of perm ${t}.`),t.forEach(e=>{f(e>=0&&e<r.rank,()=>`All entries in 'perm' must be between 0 and ${r.rank-1} but got ${t}`)}),r.rank<=1)return r.clone();let i={x:r},a={perm:t};return r.dtype===`complex64`?Vo(()=>{let e=es(r),t=Xo(r);return e=P.runKernel(rr,{x:e},a),t=P.runKernel(rr,{x:t},a),n&&(t=Qo(t)),Wi(e,t)}):P.runKernel(rr,i,a)}var ns=I({transpose_:ts});function rs(e,t){let n=e.length,r=[];for(let i=0;i<n;i++){let a=n-1-i,o=e[a]||1;(t[t.length-1-i]||1)>1&&o===1&&r.unshift(a)}return r}function is(e,t){let n=[];for(let r=0;r<t.length;r++){let i=e[e.length-r-1],a=t.length-r-1,o=t[a];(i==null||i===1&&o>1)&&n.unshift(a)}return n}function z(e,t){let n=[],r=Math.max(e.length,t.length);for(let i=0;i<r;i++){let r=e[e.length-i-1];r??=1;let a=t[t.length-i-1];if(a??=1,r===1)n.unshift(a);else if(a===1)n.unshift(r);else if(r!==a){let n=`Operands could not be broadcast together with shapes ${e} and ${t}.`;throw Error(n)}else n.unshift(r)}return n}function as(e,t,n){if(m(e),t!=null&&t.length!==3)throw Error(`tensor3d() requires shape to have three numbers`);let r=Ri(e,n);if(r.length!==3&&r.length!==1)throw Error(`tensor3d() requires values to be number[][][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor3d() requires shape to be provided when `values` are a flat array");return Gi(e,t,r,n)}function os(e,t){let n=e.shape.length,r=t.shape.length;if(n<1)throw Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${n}.`);if(r<1)throw Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${r}.`);if(t.dtype!==`int32`)throw Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.shape[r-1]>n)throw Error(`index innermost dimension length must be <= tensor rank; saw: ${t.shape[r-1]} vs. ${n}`);if(g(e.shape)===0)throw Error(`Requested more than 0 entries, but input is empty. Input shape: ${e.shape}.`);let i=t.shape,a=i[i.length-1],o=1;for(let e=0;e<i.length-1;++e)o*=i[e];let s=e.shape,c=i.slice();c.pop();let l=1;for(let e=a;e<n;++e)l*=s[e],c.push(s[e]);let u=[...j(e.shape).map(e=>e/l),1].slice(0,a);return[c,o,l,u]}function ss(e,t,n){let r=t.rank>1?t.shape[t.rank-1]:1,i=t.rank>1?t.rank-1:1,a=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${t.shape}, shape: ${e}, sliceDim: ${r}, and batchDim: ${i}.`;if(n.rank<i)throw Error(a+` update.rank < ${i}. `);if(e.length<r+(n.rank-i))throw Error(a+` Output shape length < ${r+(n.rank-i)}`);if(n.rank!==i+e.length-r)throw Error(a+` update.rank != ${i+e.length-r}`);for(let e=0;e<i;++e)if(n.shape[e]!==t.shape[e])throw Error(a+` updates.shape[${e}] (${n.shape[e]}) != indices.shape[${e}] (${t.shape[e]}).`);for(let t=0;t<n.rank-i;++t)if(n.shape[t+i]!==e[t+r])throw Error(a+` updates.shape[${t+i}] (${n.shape[t+i]}) != shape[${t+i}] (${e[t+i]})`)}function cs(e,t,n){if(t.rank<1)throw Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${e.rank}.`);if(t.dtype!==`int32`)throw Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(n.length<1)throw Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(n.length===0){if(t.size===0)throw Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(e.size===0)throw Error(`Updates specified for empty output. updates shape: ${e.shape}`)}ss(n,t,e)}function ls(e,t,n){let r=t.shape.length,i=r>1?t.shape[r-1]:1,a=n.length,o=1;for(let e=i;e<a;++e)o*=n[e];let s=i<1?1:i,c=g(t.shape)/s,l=[...j(n.slice(0,i)),1],u=g(n);return{sliceRank:i,numUpdates:c,sliceSize:o,strides:l,outputSize:u}}var us=t({assertParamsValid:()=>ps,computeFlatOffset:()=>Es,computeOutShape:()=>hs,getNormalizedAxes:()=>ys,isSliceContinous:()=>Ts,maskToAxes:()=>ms,parseSliceParams:()=>Ds,sliceInfo:()=>Os,startForAxis:()=>Cs,startIndicesWithElidedDims:()=>bs,stopForAxis:()=>ws,stopIndicesWithElidedDims:()=>xs,stridesForAxis:()=>Ss,stridesWithElidedDims:()=>gs}),ds=-2,fs=-1;function ps(e,t,n){let r=e.shape.length;f(r===t.length,()=>`Error in slice${r}D: Length of begin ${t} must match the rank of the array (${r}).`),f(r===n.length,()=>`Error in slice${r}D: Length of size ${n} must match the rank of the array (${r}).`);for(let i=0;i<r;++i)f(t[i]+n[i]<=e.shape[i],()=>`Error in slice${r}D: begin[${i}] + size[${i}] (${t[i]+n[i]}) would overflow input.shape[${i}] (${e.shape[i]})`)}function ms(e){let t=[],n=0;for(;e>0;)e&1&&t.push(n),e/=2,n++;return t}function hs(e,t,n){let r=[];for(let i=0;i<e.length;i++)r[i]=Math.ceil((t[i]-e[i])/n[i]);return r}function gs(e,t,n,r){let i=[...e];for(let e=i.length;e<r.length;e++)i.push(1);for(let e=0;e<n;e++)e===0?i[t]=1:(i.splice(t,0,1),i.pop());return i}function _s(e,t,n){return n<=e?n:n-(t-1)}function vs(e,t){let n=[];for(let r=0;r<e;r++)n.push(t+r);return n}function ys(e,t,n,r,i,a,o,s,c){let l=e.length,u=Array(l),d=Array(l),f=Array(l);if(t.length&&n>0){let c=t[0],l=n+1;u=bs(o,c,l,r,e),d=xs(s,c,l,i,e),f=gs(a,c,l,e)}else for(let t=0;t<l;t++)u[t]=Cs(o,r,a,e,t,c),d[t]=ws(s,i,a,e,t,c),f[t]=Ss(a,t,c);return{begin:u,end:d,strides:f}}function bs(e,t,n,r,i){let a=[...i],o=vs(n,t);for(let i=0;i<a.length;i++)if(o.indexOf(i)>-1)a[i]=0;else{let o=_s(t,n,i),s=r[o];e&1<<o&&(s=0),a[i]=s}return a}function xs(e,t,n,r,i){let a=[...i],o=vs(n,t);for(let i=0;i<a.length;i++)if(o.indexOf(i)>-1)a[i]=2**53-1;else{let o=_s(t,n,i),s=r[o];e&1<<o&&(s=2**53-1),a[i]=s}for(let e=0;e<a.length;e++){let t=i[e];a[e]<0&&(a[e]+=t),a[e]=c(0,a[e],i[e])}return a}function Ss(e,t,n){let r=e[t];return(n&1<<t||r==null)&&(r=1),r}function Cs(e,t,n,r,i,a){let o=t[i],s=n[i]||1;(e&1<<i||a&1<<i||o==null)&&(o=s>0?-(2**53-1):2**53-1);let l=r[i];return o<0&&(o+=l),o=c(0,o,l-1),o}function ws(e,t,n,r,i,a){let o=t[i],s=n[i]||1;(e&1<<i||a&1<<i||o==null)&&(o=s>0?2**53-1:-(2**53-1));let l=r[i];return o<0&&(o+=l),o=s>0?c(0,o,l):c(-1,o,l-1),o}function Ts(e,t,n){let r=n.length;for(let e=0;e<n.length;e++)if(n[e]>1){r=e;break}for(let i=r+1;i<n.length;i++)if(t[i]>0||n[i]!==e[i])return!1;return!0}function Es(e,t){let n=e.length>0?e[e.length-1]:1;for(let r=0;r<e.length-1;r++)n+=e[r]*t[r];return n}function Ds(e,t,n){let r,i=e.shape.length;r=typeof t==`number`?[t,...Array(i-1).fill(0)]:t.length<i?t.concat(Array(i-t.length).fill(0)):t.slice(),r.forEach(e=>{f(e!==-1,()=>`slice() does not support negative begin indexing.`)});let a;return a=n==null?Array(i).fill(-1):typeof n==`number`?[n,...Array(i-1).fill(-1)]:n.length<i?n.concat(Array(i-n.length).fill(-1)):n,a=a.map((t,n)=>t>=0?t:(f(t===-1,()=>`Negative size values should be exactly -1 but got ${t} for the slice() size at index ${n}.`),e.shape[n]-r[n])),[r,a]}function Os(e,t,n,r,i,a,o,s,c){let l;if(r==null?(l=Array(t.length),l.fill(1)):l=r,o!=null&&o&o-1)throw Error(`Multiple ellipses in slice is not allowed.`);let u=!1,d={dims:l.length,numAddAxisAfterEllipsis:0,begin:t.slice(),end:n.slice(),strides:l.slice(),beginMask:i,endMask:a,ellipsisMask:o,newAxisMask:s,shrinkAxisMask:c};for(let e=0;e<d.dims;e++)u&&1<<e&s&&d.numAddAxisAfterEllipsis++,1<<e&o&&(u=!0);u||(d.ellipsisMask|=1<<d.dims,d.dims++);let f={dims:e.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};ks(d,f);let p=!0,m=!0,h=!0,g=[],_=[];for(let t=0;t<e.length;++t){if(f.strides[t]===0)throw Error(`strides[${t}] must be non-zero`);let n=!!(f.shrinkAxisMask&1<<t),r=e[t];if(r===-1){g.push(n?1:-1);continue}let i=[f.beginMask&1<<t,f.endMask&1<<t],a=[f.strides[t]>0?0:-1,f.strides[t]>0?r:r-1];if(n&&f.strides[t]<=0)throw Error(`only stride 1 allowed on non-range indexing.`);h&&=f.strides[t]===1;let o=!!(f.beginMask&1<<t&&f.endMask&1<<t);if(f.beginValid&&f.endValid){if(n){let e=f.begin[t]<0?r+f.begin[t]:f.begin[t];if(f.begin[t]=e,f.end[t]=f.begin[t]+1,e<0||e>=r)throw Error(`slice index ${f.begin[t]} of dimension ${t} out of bounds.`)}else f.begin[t]=As(f.begin[t],0,f.strides[t],r,i,a),f.end[t]=As(f.end[t],1,f.strides[t],r,i,a);let e=f.strides[t]===1&&f.begin[t]===0&&f.end[t]===r;p&&=e,m&&=t===0&&f.strides[t]===1||e}else p=p&&f.strides[t]===1&&o,m&&=t===0&&f.strides[t]===1||o;let s,c=!1;if(f.beginValid&&f.endValid?(s=f.end[t]-f.begin[t],c=!0):n?(s=1,c=!0):o&&r>=0&&(s=f.strides[t]<0?-r:r,c=!0),c){let e;e=s===0||s<0!=f.strides[t]<0?0:Math.trunc(s/f.strides[t])+(s%f.strides[t]===0?0:1),g.push(e)}else g.push(-1)}for(let e=0;e<f.finalShapeGatherIndices.length;++e){let t=f.finalShapeGatherIndices[e];t>=0?_.push(g[t]):t===ds&&_.push(1)}return{finalShapeSparse:_.filter((e,t)=>f.finalShapeGatherIndices[t]!==ds),finalShape:_,isIdentity:p,sliceDim0:m,isSimpleSlice:h,begin:f.begin,end:f.end,strides:f.strides}}function ks(e,t){t.beginMask=0,t.endMask=0,t.shrinkAxisMask=0;let n=0;t.beginValid=e.begin!=null,t.endValid=e.end!=null,t.begin=Array(t.dims),t.end=Array(t.dims),t.strides=Array(t.dims),t.finalShapeGatherIndices=[],t.finalShapeGatherIndicesSparse=[],t.inputShapeGatherIndicesSparse=Array(t.dims);for(let r=0;r<e.dims;r++)if(1<<r&e.ellipsisMask){let i=Math.min(t.dims-(e.dims-r)+1+e.numAddAxisAfterEllipsis,t.dims);for(;n<i;n++)t.begin[n]=0,t.end[n]=0,t.strides[n]=1,t.beginMask|=1<<n,t.endMask|=1<<n,t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(-1),t.inputShapeGatherIndicesSparse[n]=r}else if(1<<r&e.newAxisMask)t.finalShapeGatherIndices.push(ds),t.finalShapeGatherIndicesSparse.push(-1);else{if(n===t.begin.length)throw Error(`Index out of range using input dim ${n}; input has only ${t.dims} dims, ${t.begin.length}.`);e.begin!=null&&(t.begin[n]=e.begin[r]),e.end!=null&&(t.end[n]=e.end[r]),t.strides[n]=e.strides[r],e.beginMask&1<<r&&(t.beginMask|=1<<n),e.endMask&1<<r&&(t.endMask|=1<<n),e.shrinkAxisMask&1<<r?(t.finalShapeGatherIndices.push(fs),t.finalShapeGatherIndicesSparse.push(-1),t.shrinkAxisMask|=1<<n):(t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(r)),t.inputShapeGatherIndicesSparse[n]=r,n++}}function As(e,t,n,r,i,a){if(i[t])return n>0?a[t]:a[t+1&1];{let t=e<0?r+e:e;return t<a[0]?a[0]:t>a[1]?a[1]:t}}function js(e,t){let n=F(e,`a`,`add`),r=F(t,`b`,`add`);[n,r]=Ci(n,r);let i={a:n,b:r};return P.runKernel(`Add`,i)}var Ms=I({add_:js});function Ns(e,t){let n=F(e,`a`,`floorDiv`),r=F(t,`b`,`floorDiv`);[n,r]=Ci(n,r);let i={a:n,b:r};return P.runKernel(Tt,i)}var Ps=I({floorDiv_:Ns});function Fs(e,t){let n=F(e,`a`,`div`),r=F(t,`b`,`div`);if([n,r]=Ci(n,r),n.dtype===`int32`&&r.dtype===`int32`)return Ps(n,r);let i={a:n,b:r};return P.runKernel(gt,i,{})}var Is=I({div_:Fs});function Ls(e,t){let n=F(e,`a`,`mul`),r=F(t,`b`,`mul`);[n,r]=Ci(n,r);let i={a:n,b:r};return P.runKernel(nn,i)}var B=I({mul_:Ls});function Rs(e){let t=F(e,`x`,`abs`);if(t.dtype===`complex64`){let e={x:t};return P.runKernel(Ye,e)}{let e={x:t};return P.runKernel(`Abs`,e)}}var zs=I({abs_:Rs});function Bs(e){let t={x:F(e,`x`,`acos`)};return P.runKernel(De,t)}var Vs=I({acos_:Bs});function Hs(e){let t={x:F(e,`x`,`acosh`)};return P.runKernel(Oe,t)}var Us=I({acosh_:Hs});function Ws(e){f(Array.isArray(e),()=>`The argument passed to tf.addN() must be a list of tensors`),f(e.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${e.length}`);let t=e.map((e,t)=>F(e,`tensors${t}`,`addN`)),n=t[0];t.forEach(e=>{if(e.dtype!==n.dtype)throw Error(`All tensors passed to tf.addN() must have the same dtype`)}),t.forEach(e=>{if(!_(e.shape,n.shape))throw Error(`All tensors passed to tf.addN() must have the same shape`)});let r=t;return P.runKernel(ke,r)}var Gs=I({addN_:Ws});function Ks(e,t=null,n=!1){let r={x:F(e,`x`,`all`,`bool`)},i={axis:t,keepDims:n};return P.runKernel(`All`,r,i)}var qs=I({all_:Ks});function Js(e,t=null,n=!1){let r={x:F(e,`x`,`any`,`bool`)},i={axis:t,keepDims:n};return P.runKernel(`Any`,r,i)}var Ys=I({any_:Js});function Xs(e,t=0){let n={x:F(e,`x`,`argMax`)},r={axis:t};return P.runKernel(Ae,n,r)}var Zs=I({argMax_:Xs});function Qs(e,t=0){let n={x:F(e,`x`,`argMin`)},r={axis:t};return P.runKernel(je,n,r)}var $s=I({argMin_:Qs});function ec(e){let t={x:F(e,`x`,`asin`)};return P.runKernel(Me,t)}var tc=I({asin_:ec});function nc(e){let t={x:F(e,`x`,`asinh`)};return P.runKernel(Ne,t)}var rc=I({asinh_:nc});function ic(e){let t={x:F(e,`x`,`atan`)};return P.runKernel(Pe,t)}var ac=I({atan_:ic});function oc(e,t){let n=F(e,`a`,`atan2`),r=F(t,`b`,`atan2`);[n,r]=Ci(n,r);let i={a:n,b:r};return P.runKernel(Ie,i)}var sc=I({atan2_:oc});function cc(e){let t={x:F(e,`x`,`atanh`)};return P.runKernel(Fe,t)}var lc=I({atanh_:cc});function uc(e,t,n,r,i=`NHWC`,a){let o=e[3];return pc(e,[...t,o],n,a,r,null,null,Ec(i))}function dc(e,t,n,r,i,a,o=`channelsLast`){let[s,c]=vc(t),l;if(o===`channelsLast`)l=[s,c,e[3],e[3]];else if(o===`channelsFirst`)l=[s,c,e[1],e[1]];else throw Error(`Unknown dataFormat ${o}`);return pc(e,l,n,r,i,a,!1,o)}function fc(e,t,n,r,i,a,o=`NDHWC`){let[s,c,l]=yc(t),u,d;if(o===`NDHWC`)d=`channelsLast`,u=[s,c,l,e[4],e[4]];else if(o===`NCDHW`)d=`channelsFirst`,u=[s,c,l,e[1],e[1]];else throw Error(`Unknown dataFormat ${o}`);return mc(e,u,n,r,i,!1,d,a)}function pc(e,t,n,r,i,a,o=!1,s=`channelsLast`){let[c,l,u,d]=[-1,-1,-1,-1];if(s===`channelsLast`)[c,l,u,d]=e;else if(s===`channelsFirst`)[c,d,l,u]=e;else throw Error(`Unknown dataFormat ${s}`);let[f,p,,m]=t,[h,g]=vc(n),[_,v]=vc(r),y=bc(f,_),b=bc(p,v),{padInfo:x,outHeight:S,outWidth:C}=xc(i,l,u,h,g,y,b,a,s),w=o?m*d:m,T;return s===`channelsFirst`?T=[c,w,S,C]:s===`channelsLast`&&(T=[c,S,C,w]),{batchSize:c,dataFormat:s,inHeight:l,inWidth:u,inChannels:d,outHeight:S,outWidth:C,outChannels:w,padInfo:x,strideHeight:h,strideWidth:g,filterHeight:f,filterWidth:p,effectiveFilterHeight:y,effectiveFilterWidth:b,dilationHeight:_,dilationWidth:v,inShape:e,outShape:T,filterShape:t}}function mc(e,t,n,r,i,a=!1,o=`channelsLast`,s){let[c,l,u,d,f]=[-1,-1,-1,-1,-1];if(o===`channelsLast`)[c,l,u,d,f]=e;else if(o===`channelsFirst`)[c,f,l,u,d]=e;else throw Error(`Unknown dataFormat ${o}`);let[p,m,h,,g]=t,[_,v,y]=yc(n),[b,x,S]=yc(r),C=bc(p,b),w=bc(m,x),T=bc(h,S),{padInfo:E,outDepth:D,outHeight:O,outWidth:ee}=Sc(i,l,u,d,_,v,y,C,w,T,s),k=a?g*f:g,A;return o===`channelsFirst`?A=[c,k,D,O,ee]:o===`channelsLast`&&(A=[c,D,O,ee,k]),{batchSize:c,dataFormat:o,inDepth:l,inHeight:u,inWidth:d,inChannels:f,outDepth:D,outHeight:O,outWidth:ee,outChannels:k,padInfo:E,strideDepth:_,strideHeight:v,strideWidth:y,filterDepth:p,filterHeight:m,filterWidth:h,effectiveFilterDepth:C,effectiveFilterHeight:w,effectiveFilterWidth:T,dilationDepth:b,dilationHeight:x,dilationWidth:S,inShape:e,outShape:A,filterShape:t}}function hc(e,t,n,r,i){r??=_c(e,t,n);let a=e[0],o=e[1];return[Cc((a-t+2*r)/n+1,i),Cc((o-t+2*r)/n+1,i)]}function gc(e,t,n,r,i,a){i??=_c(e,t,r);let o=e[0],s=e[1],c=e[2];return[Cc((o-t+2*i)/r+1,a),Cc((s-t+2*i)/r+1,a),Cc((c-t+2*i)/r+1,a),n]}function _c(e,t,n,r=1){let i=bc(t,r);return Math.floor((e[0]*(n-1)-n+i)/2)}function vc(e){return typeof e==`number`?[e,e,e]:e.length===2?[e[0],e[1],1]:e}function yc(e){return typeof e==`number`?[e,e,e]:e}function bc(e,t){return t<=1?e:e+(e-1)*(t-1)}function xc(e,t,n,r,i,a,o,s,c){let l,u,d;if(typeof e==`number`){l={top:e,bottom:e,left:e,right:e,type:e===0?`VALID`:`NUMBER`};let i=hc([t,n],a,r,e,s);u=i[0],d=i[1]}else if(e===`same`){u=Math.ceil(t/r),d=Math.ceil(n/i);let e=Math.max(0,(u-1)*r+a-t),s=Math.max(0,(d-1)*i+o-n),c=Math.floor(e/2),f=e-c,p=Math.floor(s/2);l={top:c,bottom:f,left:p,right:s-p,type:`SAME`}}else if(e===`valid`)l={top:0,bottom:0,left:0,right:0,type:`VALID`},u=Math.ceil((t-a+1)/r),d=Math.ceil((n-o+1)/i);else if(typeof e==`object`){let f=c===`channelsLast`?e[1][0]:e[2][0],p=c===`channelsLast`?e[1][1]:e[2][1],m=c===`channelsLast`?e[2][0]:e[3][0],h=c===`channelsLast`?e[2][1]:e[3][1];l={top:f,bottom:p,left:m,right:h,type:f===0&&p===0&&m===0&&h===0?`VALID`:`EXPLICIT`},u=Cc((t-a+f+p)/r+1,s),d=Cc((n-o+m+h)/i+1,s)}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:l,outHeight:u,outWidth:d}}function Sc(e,t,n,r,i,a,o,s,c,l,u){let d,f,p,m;if(typeof e==`number`){d={top:e,bottom:e,left:e,right:e,front:e,back:e,type:e===0?`VALID`:`NUMBER`};let a=gc([t,n,r,1],s,1,i,e,u);f=a[0],p=a[1],m=a[2]}else if(e===`same`){f=Math.ceil(t/i),p=Math.ceil(n/a),m=Math.ceil(r/o);let e=(f-1)*i+s-t,u=(p-1)*a+c-n,h=(m-1)*o+l-r,g=Math.floor(e/2),_=e-g,v=Math.floor(u/2),y=u-v,b=Math.floor(h/2);d={top:v,bottom:y,left:b,right:h-b,front:g,back:_,type:`SAME`}}else if(e===`valid`)d={top:0,bottom:0,left:0,right:0,front:0,back:0,type:`VALID`},f=Math.ceil((t-s+1)/i),p=Math.ceil((n-c+1)/a),m=Math.ceil((r-l+1)/o);else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:d,outDepth:f,outHeight:p,outWidth:m}}function Cc(e,t){if(!t)return Math.trunc(e);switch(t){case`round`:return Math.round(e);case`ceil`:return Math.ceil(e);case`floor`:return Math.floor(e);default:throw Error(`Unknown roundingMode ${t}`)}}function wc(e){let[t,n,r]=vc(e);return t===1&&n===1&&r===1}function Tc(e,t){return wc(e)||wc(t)}function Ec(e){if(e===`NHWC`)return`channelsLast`;if(e===`NCHW`)return`channelsFirst`;throw Error(`Unknown dataFormat ${e}`)}function Dc(e,t,n){if(n!=null){if(typeof t==`string`)throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if(typeof t==`number`)f(v(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else if(typeof t==`object`)t.forEach(t=>{t.forEach(t=>{f(v(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`)})});else throw Error(`Error in ${e}: Unknown padding parameter: ${t}`)}}function Oc(e,t){let n={x:F(e,`x`,`reshape`,`string_or_numeric`)},r={shape:t};return P.runKernel(bn,n,r)}var V=I({reshape_:Oc});function kc(e,t,n,r,i){let a=F(e,`x`,`avgPool`,`float32`);f(Tc(n,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`);let o=a,s=!1;a.rank===3&&(s=!0,o=V(a,[1,a.shape[0],a.shape[1],a.shape[2]])),f(o.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${o.rank}.`),Dc(`avgPool`,r,i);let c={x:o},l={filterSize:t,strides:n,pad:r,dimRoundingMode:i},u=P.runKernel(Le,c,l);return u=io(u,a.dtype),s?V(u,[u.shape[1],u.shape[2],u.shape[3]]):u}var Ac=I({avgPool_:kc});function jc(e,t,n,r,i,a=`NDHWC`){let o=F(e,`x`,`avgPool3d`,`float32`),s=o,c=!1;o.rank===4&&(c=!0,s=V(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),f(s.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${s.rank}.`),f(a===`NDHWC`,()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),Dc(`avgPool3d`,r,i);let l={x:s},u={filterSize:t,strides:n,pad:r,dimRoundingMode:i,dataFormat:a},d=P.runKernel(ze,l,u);return d=io(d,s.dtype),c?V(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}var Mc=I({avgPool3d_:jc});function Nc(e,t=0){f(e.length>=1,()=>`Pass at least one tensor to concat`);let n=Vi(e,`tensors`,`concat`,`string_or_numeric`);if(n[0].dtype===`complex64`&&n.forEach(e=>{if(e.dtype!==`complex64`)throw Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${e.dtype}. `)}),n.length===1)return oo(n[0]);let r=n,i={axis:t};return P.runKernel(Xe,r,i)}var Pc=I({concat_:Nc});function Fc(e){let t={x:F(e,`x`,`sigmoid`,`float32`)};return P.runKernel(In,t)}var Ic=I({sigmoid_:Fc});function Lc(e,t,n){let r=F(e,`x`,`slice`,`string_or_numeric`);if(r.rank===0)throw Error(`Slicing scalar is not possible`);let i={x:r},a={begin:t,size:n};return P.runKernel(Nn,i,a)}var H=I({slice_:Lc});function Rc(e){let t={x:F(e,`x`,`tanh`,`float32`)};return P.runKernel($n,t)}var zc=I({tanh_:Rc});function Bc(e,t,n,r,i,a){let o=F(e,`forgetBias`,`basicLSTMCell`),s=F(t,`lstmKernel`,`basicLSTMCell`),c=F(n,`lstmBias`,`basicLSTMCell`),l=F(r,`data`,`basicLSTMCell`),u=F(i,`c`,`basicLSTMCell`),d=Ms(R(Pc([l,F(a,`h`,`basicLSTMCell`)],1),s),c),f=d.shape[0],p=d.shape[1]/4,m=[f,p],h=H(d,[0,0],m),g=H(d,[0,p],m),_=H(d,[0,p*2],m),v=H(d,[0,p*3],m),y=Ms(B(Ic(h),zc(g)),B(u,Ic(Ms(o,_))));return[y,B(zc(y),Ic(v))]}var Vc=I({basicLSTMCell_:Bc});function Hc(e,t,n){let r=F(e,`x`,`batchToSpaceND`),i=t.reduce((e,t)=>e*t);f(r.rank>=1+t.length,()=>`input rank is ${r.rank} but should be > than blockShape.length ${t.length}`),f(n.length===t.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${t.length}`),f(r.shape[0]%i===0,()=>`input tensor batch is ${r.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(` * `)} === ${i}`);let a={x:r},o={blockShape:t,crops:n};return P.runKernel(He,a,o)}var Uc=I({batchToSpaceND_:Hc});function Wc(e){let t;return t=e.rank===0||e.rank===1?V(e,[1,1,1,e.size]):e.rank===2?V(e,[1,1,e.shape[0],e.shape[1]]):e.rank===3?V(e,[1,e.shape[0],e.shape[1],e.shape[2]]):e,t}function Gc(e,t,n,r,i,a){a??=.001;let o=F(e,`x`,`batchNorm`),s=F(t,`mean`,`batchNorm`),c=F(n,`variance`,`batchNorm`),l;i!=null&&(l=F(i,`scale`,`batchNorm`));let u;r!=null&&(u=F(r,`offset`,`batchNorm`)),f(s.rank===c.rank,()=>`Batch normalization gradient requires mean and variance to have equal ranks.`),f(u==null||s.rank===u.rank,()=>`Batch normalization gradient requires mean and offset to have equal ranks.`),f(l==null||s.rank===l.rank,()=>`Batch normalization gradient requires mean and scale to have equal ranks.`);let d={x:Wc(o),scale:l,offset:u,mean:s,variance:c},p={varianceEpsilon:a};return V(P.runKernel(Et,d,p),o.shape)}var Kc=I({batchNorm_:Gc});function qc(e,t,n,r,i,a){let o=F(e,`x`,`batchNorm`),s=F(t,`mean`,`batchNorm`),c=F(n,`variance`,`batchNorm`),l;i!=null&&(l=F(i,`scale`,`batchNorm`));let u;return r!=null&&(u=F(r,`offset`,`batchNorm`)),f(o.rank===2,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${o.rank}.`),f(s.rank===2||s.rank===1,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${s.rank}.`),f(c.rank===2||c.rank===1,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${c.rank}.`),l!=null&&f(l.rank===2||l.rank===1,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${l.rank}.`),u!=null&&f(u.rank===2||u.rank===1,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${u.rank}.`),Kc(o,s,c,u,l,a)}var Jc=I({batchNorm2d_:qc});function Yc(e,t,n,r,i,a){let o=F(e,`x`,`batchNorm`),s=F(t,`mean`,`batchNorm`),c=F(n,`variance`,`batchNorm`),l;i!=null&&(l=F(i,`scale`,`batchNorm`));let u;return r!=null&&(u=F(r,`offset`,`batchNorm`)),f(o.rank===3,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${o.rank}.`),f(s.rank===3||s.rank===1,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${s.rank}.`),f(c.rank===3||c.rank===1,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${c.rank}.`),l!=null&&f(l.rank===3||l.rank===1,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${l.rank}.`),u!=null&&f(u.rank===3||u.rank===1,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${u.rank}.`),Kc(o,s,c,u,l,a)}var Xc=I({batchNorm3d_:Yc});function Zc(e,t,n,r,i,a){let o=F(e,`x`,`batchNorm`),s=F(t,`mean`,`batchNorm`),c=F(n,`variance`,`batchNorm`),l;i!=null&&(l=F(i,`scale`,`batchNorm`));let u;return r!=null&&(u=F(r,`offset`,`batchNorm`)),f(o.rank===4,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${o.rank}.`),f(s.rank===4||s.rank===1,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${s.rank}.`),f(c.rank===4||c.rank===1,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${c.rank}.`),l!=null&&f(l.rank===4||l.rank===1,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${l.rank}.`),u!=null&&f(u.rank===4||u.rank===1,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${u.rank}.`),Kc(o,s,c,u,l,a)}var Qc=I({batchNorm4d_:Zc});function $c(e,t,n){let r=F(e,`x`,`bincount`),i=F(t,`weights`,`bincount`);f(r.dtype===`int32`,()=>`Error in bincount: input dtype must be int32, but got ${r.dtype}`),f(n>=0,()=>`size must be non-negative, but got ${n}.`),f(i.size===r.size||i.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${r.shape}, weights shape: ${i.shape}.`);let a={x:r,weights:i},o={size:n};return P.runKernel(Ue,a,o)}var el=I({bincount_:$c});function tl(e,t){let n=F(e,`s0`,`broadcastArgs`,`int32`),r=F(t,`s1`,`broadcastArgs`,`int32`);if(n.rank!==1)throw Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${n.rank}`);if(r.rank!==1)throw Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${r.rank}`);let i={s0:n,s1:r};return P.runKernel(We,i)}var nl=I({broadcastArgs_:tl});function rl(e,t){let n=F(e,`broadcastTo`,`x`),r=n.shape;if(t.some(e=>!(e>0)||e%1!=0))throw Error(`broadcastTo(): Invalid broadcast shape [${t}].`);if(t.length<n.rank)throw Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){let e=n.shape.slice();for(;e.length<t.length;)e.unshift(1);n=V(n,e)}let i=n.shape,a=Array.from(t);for(let e=t.length-1;e>=0;e--)if(i[e]===t[e])a[e]=1;else if(n.shape[e]!==1)throw Error(`broadcastTo(): [${r}] cannot be broadcast to [${t}].`);if(a.map((e,t)=>e>1?t:-1).filter(e=>e>=0).length===0)return oo(n);let o={x:n},s={reps:a};return P.runKernel(er,o,s)}var il=I({broadcastTo_:rl});function al(e){let t={x:F(e,`x`,`ceil`,`float32`)};return P.runKernel(Ke,t)}var ol=I({ceil_:al});function sl(e,t,n){let r={shape:e,value:t,dtype:n};return P.runKernel(St,{},r)}function cl(e,t,n){let r=F(e,`x`,`clipByValue`);if(f(t<=n,()=>`Error in clip: min (${t}) must be less than or equal to max (${n}).`),t===n)return sl(r.shape,t,r.dtype);let i={x:r},a={clipValueMin:t,clipValueMax:n};return P.runKernel(qe,i,a)}var ll=I({clipByValue_:cl});function ul(e){return Pc(e,0)}var dl=I({concat1d_:ul});function fl(e,t){return Pc(e,t)}var pl=I({concat2d_:fl});function ml(e,t){return Pc(e,t)}var hl=I({concat3d_:ml});function gl(e,t){return Pc(e,t)}var _l=I({concat4d_:gl});function vl(e,t,n,r,i=`NHWC`,a=[1,1],o){let s=F(e,`x`,`conv2d`,`float32`),c=F(t,`filter`,`conv2d`,`float32`),l=s,u=!1;s.rank===3&&(u=!0,l=V(s,[1,s.shape[0],s.shape[1],s.shape[2]])),f(l.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${l.rank}.`),f(c.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${c.rank}.`),Dc(`conv2d`,r,o);let d=i===`NHWC`?l.shape[3]:l.shape[1];f(d===c.shape[2],()=>`Error in conv2d: depth of input (${d}) must match input depth for filter ${c.shape[2]}.`),f(Tc(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`);let p={x:l,filter:c},m={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o},h=P.runKernel(Ze,p,m);return u?V(h,[h.shape[1],h.shape[2],h.shape[3]]):h}var yl=I({conv2d_:vl});function bl(e,t,n,r,i=`NWC`,a=1,o){let s=F(e,`x`,`conv1d`),c=F(t,`filter`,`conv1d`),l=s,u=!1;s.rank===2&&(u=!0,l=V(s,[1,s.shape[0],s.shape[1]])),f(l.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${l.rank}.`),f(c.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${c.rank}.`),Dc(`conv1d`,r,o),f(l.shape[2]===c.shape[1],()=>`Error in conv1d: depth of input (${l.shape[2]}) must match input depth for filter ${c.shape[1]}.`),f(Tc(n,a),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${a}'`),f(i===`NWC`,()=>`Error in conv1d: got dataFormat of ${i} but only NWC is currently supported.`);let d=V(c,[1,c.shape[0],c.shape[1],c.shape[2]]),p=yl(V(l,[l.shape[0],1,l.shape[1],l.shape[2]]),d,[1,n],r,`NHWC`,[1,a],o);return u?V(p,[p.shape[2],p.shape[3]]):V(p,[p.shape[0],p.shape[2],p.shape[3]])}var xl=I({conv1d_:bl});function Sl(e,t,n,r,i,a=`NHWC`,o){f(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let s=e,c=t,l=!1;t.rank===3&&(l=!0,c=V(t,[1,t.shape[0],t.shape[1],t.shape[2]]),s=[1,e[0],e[1],e[2]]),f(s.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${s.length}.`),f(c.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${c.rank}`),f(n.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${n.rank}`);let u=a===`NHWC`?s[3]:s[1],d=a===`NHWC`?c.shape[3]:c.shape[1];f(u===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${u}) must match input depth for filter ${n.shape[2]}.`),f(d===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${d}) must match output depth for filter ${n.shape[3]}.`),Dc(`conv2dDerInput`,i,o);let p={dy:c,filter:n},m={strides:r,pad:i,dataFormat:a,dimRoundingMode:o,inputShape:s},h=P.runKernel($e,p,m);return l?V(h,[h.shape[1],h.shape[2],h.shape[3]]):h}var Cl=I({conv2DBackpropInput_:Sl});function wl(e,t,n,r,i,a){return Cl(n,F(e,`x`,`conv2dTranspose`),F(t,`filter`,`conv2dTranspose`),r,i,`NHWC`,a)}var Tl=I({conv2dTranspose_:wl});function El(e,t,n,r,i=`NDHWC`,a=[1,1,1]){let o=F(e,`x`,`conv3d`),s=F(t,`filter`,`conv3d`),c=o,l=!1;o.rank===4&&(l=!0,c=V(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),f(c.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${c.rank}.`),f(s.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${s.rank}.`),f(c.shape[4]===s.shape[3],()=>`Error in conv3d: depth of input (${c.shape[4]}) must match input depth for filter ${s.shape[3]}.`),f(Tc(n,a),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),f(i===`NDHWC`,()=>`Error in conv3d: got dataFormat of ${i} but only NDHWC is currently supported.`);let u={x:c,filter:s},d={strides:n,pad:r,dataFormat:i,dilations:a},p=P.runKernel(et,u,d);return l?V(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}var Dl=I({conv3d_:El});function Ol(e,t,n,r,i){f(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let a=e,o=t,s=!1;t.rank===4&&(s=!0,o=V(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),a=[1,e[0],e[1],e[2],e[3]]);let c=a[4],l=o.shape[4];f(a.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${a.length}.`),f(o.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${o.rank}`),f(n.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${n.rank}`),f(c===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${c}) must match input depth for filter ${n.shape[3]}.`),f(l===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${l}) must match output depth for filter ${n.shape[4]}.`);let u={dy:o,filter:n},d={pad:i,strides:r,inputShape:a},p=P.runKernel(nt,u,d);return s?V(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}var kl=I({conv3DBackpropInput_:Ol});function Al(e,t,n,r,i){return kl(n,F(e,`x`,`conv3dTranspose`),F(t,`filter`,`conv3dTranspose`),r,i)}var jl=I({conv3dTranspose_:Al});function Ml(e){let t={x:F(e,`x`,`cos`,`float32`)};return P.runKernel(`Cos`,t)}var Nl=I({cos_:Ml});function Pl(e){let t={x:F(e,`x`,`cosh`,`float32`)};return P.runKernel(rt,t)}var Fl=I({cosh_:Pl});function Il(e,t=0,n=!1,r=!1){let i={x:F(e,`x`,`cumprod`)},a={axis:t,exclusive:n,reverse:r};return P.runKernel(it,i,a)}var Ll=I({cumprod_:Il});function Rl(e,t=0,n=!1,r=!1){let i={x:F(e,`x`,`cumsum`)},a={axis:t,exclusive:n,reverse:r};return P.runKernel(at,i,a)}var zl=I({cumsum_:Rl});function Bl(e,t,n,r=!1){let i=F(e,`x`,`denseBincount`),a=F(t,`weights`,`denseBincount`);f(i.dtype===`int32`,()=>`Error in denseBincount: input dtype must be int32, but got ${i.dtype}`),f(i.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${i.rank}.`),f(n>=0,()=>`size must be non-negative, but got ${n}.`),f(a.size===i.size||a.size===0,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${i.shape}, weights shape: ${a.shape}.`);let o={x:i,weights:a},s={size:n,binaryOutput:r};return P.runKernel(st,o,s)}var Vl=I({denseBincount_:Bl});function Hl(e,t,n=`NHWC`){let r=F(e,`x`,`depthToSpace`,`float32`),i=n===`NHWC`?r.shape[1]:r.shape[2],a=n===`NHWC`?r.shape[2]:r.shape[3],o=n===`NHWC`?r.shape[3]:r.shape[1];f(t>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${t}`),f(i*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${i} and ${t}  for depthToSpace with input shape
    ${r.shape}`),f(a*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${a} and ${t} for depthToSpace with input shape
        ${r.shape}`),f(o%(t*t)===0,()=>`Dimension size must be evenly divisible by ${t*t} but is ${o} for depthToSpace with input shape ${r.shape}`);let s={x:r},c={blockSize:t,dataFormat:n};return P.runKernel(ct,s,c)}var Ul=I({depthToSpace_:Hl});function Wl(e,t,n,r,i=`NHWC`,a=[1,1],o){let s=F(e,`x`,`depthwiseConv2d`,`float32`),c=F(t,`filter`,`depthwiseConv2d`,`float32`),l=s,u=!1;s.rank===3&&(u=!0,l=V(s,[1,s.shape[0],s.shape[1],s.shape[2]])),f(l.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${l.rank}.`),f(c.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${c.rank}.`);let d=i===`NHWC`?l.shape[3]:l.shape[1];f(d===c.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${d}) must match the inChannels dimension in filter ${c.shape[2]}.`),Dc(`depthwiseConv2d`,r,o);let p={x:l,filter:c},m={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o},h=P.runKernel(lt,p,m);return u?V(h,[h.shape[1],h.shape[2],h.shape[3]]):h}var Gl=I({depthwiseConv2d_:Wl});function Kl(e){let t={x:F(e,`x`,`diag`)};return P.runKernel(ft,t)}var ql=I({diag_:Kl});function Jl(e,t,n,r,i=[1,1],a=`NHWC`){let o=F(e,`x`,`dilation2d`),s=F(t,`filter`,`dilation2d`);f(o.rank===3||o.rank===4,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${o.rank}.`),f(s.rank===3,()=>`Error in dilation2d: filter must be rank 3, but got rank ${s.rank}.`),f(a===`NHWC`,()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${a}`);let c=o,l=!1;o.rank===3&&(c=V(o,[1,o.shape[0],o.shape[1],o.shape[2]]),l=!0);let u={x:c,filter:s},d={strides:n,pad:r,dilations:i},p=P.runKernel(pt,u,d);return l?V(p,[p.shape[1],p.shape[2],p.shape[3]]):p}var Yl=I({dilation2d_:Jl});function Xl(e,t){let n=F(e,`a`,`equal`,`string_or_numeric`),r=F(t,`b`,`equal`,`string_or_numeric`);[n,r]=Ci(n,r),z(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(yt,i)}var Zl=I({equal_:Xl});function Ql(e,t,n){let r=F(t,`a`,`where`),i=F(n,`b`,`where`),a=F(e,`condition`,`where`,`bool`),o=z(z(a.shape,r.shape),i.shape),s={condition:il(a,o),t:il(r,o),e:il(i,o)};return P.runKernel(jn,s)}var $l=I({where_:Ql});function eu(e){let t={x:F(e,`x`,`zerosLike`)};return P.runKernel(sr,t)}var tu=I({zerosLike_:eu});function nu(e,t){let n=F(e,`a`,`div`),r=F(t,`b`,`div`);[n,r]=Ci(n,r);let i=Is(n,r),a=tu(i);return $l(Zl(r,a),a,i)}var ru=I({divNoNan_:nu});function iu(e,t){let n=F(e,`t1`,`dot`),r=F(t,`t2`,`dot`);f((n.rank===1||n.rank===2)&&(r.rank===1||r.rank===2),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${n.rank} and ${r.rank}.`);let i=n.rank===1?n.size:n.shape[1],a=r.rank===1?r.size:r.shape[0];if(f(i===a,()=>`Error in dot: inner dimensions of inputs must match, but got ${i} and ${a}.`),n.rank===1&&r.rank===1)return V(R(V(n,[1,-1]),V(r,[-1,1])),[]);if(n.rank===1&&r.rank===2){let e=R(V(n,[1,-1]),V(r,[r.shape[0],r.shape[1]]));return V(e,[e.size])}if(n.rank===2&&r.rank===1){let e=R(n,V(r,[-1,1]));return V(e,[e.size])}return R(n,V(r,[r.shape[0],r.shape[1]]))}var au=I({dot_:iu});function ou(e,...t){let n=t.map((e,t)=>F(e,`tensors${t}`,`einsum`)),r={equation:e};return P.runKernel(_t,n,r)}var su=I({einsum_:ou});function cu(e){let t={x:F(e,`x`,`elu`,`float32`)};return P.runKernel(`Elu`,t)}var lu=I({elu_:cu});function uu(e){let t=F(e,`x`,`erf`);f(t.dtype===`int32`||t.dtype===`float32`,()=>"Input dtype must be `int32` or `float32`."),t.dtype===`int32`&&(t=io(t,`float32`));let n={x:t};return P.runKernel(`Erf`,n)}var du=I({erf_:uu});function fu(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function pu(e,t,n){let r=e.length+t.length,i=[],a=0,o=0;for(let s=0;s<r;s++)n.indexOf(s)===-1?i.push(e[a++]):i.push(t[o++]);return i}function mu(e,t){let n=[],r=e.length;for(let i=0;i<r;i++)t.indexOf(i)===-1&&n.push(e[i]);return[n,t.map(t=>e[t])]}function hu(e,t){return pu(e,t.map(e=>1),t)}function gu(e,t,n){f(fu(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function _u(e,t){if(fu(e,t))return null;let n=[];for(let r=0;r<t;++r)e.indexOf(r)===-1&&n.push(r);return e.forEach(e=>n.push(e)),n}function vu(e){return e.map((e,t)=>[t,e]).sort((e,t)=>e[1]-t[1]).map(e=>e[0])}function yu(e,t){let n=[];for(let r=t-e;r<t;++r)n.push(r);return n}function bu(e,t=null,n=!1){let r={x:F(e,`x`,`max`)},i={reductionIndices:t,keepDims:n};return P.runKernel(`Max`,r,i)}var xu=I({max_:bu});function Su(e,t=null,n=!1){let r={x:F(e,`x`,`min`)},i={axis:t,keepDims:n};return P.runKernel(`Min`,r,i)}var Cu=I({min_:Su});function wu(e,t){let n=F(e,`base`,`pow`),r=F(t,`exp`,`pow`);[n,r]=Ci(n,r);let i={a:n,b:r};return P.runKernel(`Pow`,i)}var Tu=I({pow_:wu});function U(e,t){if((k(e)&&t!==`string`||Array.isArray(e))&&t!==`complex64`)throw Error(`Error creating a new Scalar: value must be a primitive (number|boolean|string)`);if(t===`string`&&k(e)&&!(e instanceof Uint8Array))throw Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return Gi(e,[],[],t)}function Eu(e){let t={x:F(e,`x`,`sqrt`,`float32`)};return P.runKernel(Rn,t)}var Du=I({sqrt_:Eu});function Ou(e){let t=F(e,`x`,`square`);return P.runKernel(`Square`,{x:t},{})}var ku=I({square_:Ou});function Au(e,t=null,n=!1){let r=F(e,`x`,`sum`);r.dtype===`bool`&&(r=io(r,`int32`));let i={x:r},a={axis:t,keepDims:n};return P.runKernel(`Sum`,i,a)}var ju=I({sum_:Au});function Mu(e,t=`euclidean`,n=null,r=!1){e=F(e,`x`,`norm`);let i=Nu(e,t,n),a=i.shape;if(r){let t=C(n,e.shape);a=hu(i.shape,t)}return V(i,a)}function Nu(e,t,n=null){if(e.rank===0)return zs(e);if(e.rank!==1&&n===null)return Nu(V(e,[-1]),t,n);if(e.rank===1||typeof n==`number`||Array.isArray(n)&&n.length===1){if(t===1)return ju(zs(e),n);if(t===1/0)return xu(zs(e),n);if(t===-1/0)return Cu(zs(e),n);if(t===`euclidean`||t===2)return Du(ju(Tu(zs(e),U(2,`int32`)),n));throw Error(`Error in norm: invalid ord value: ${t}`)}if(Array.isArray(n)&&n.length===2){if(t===1)return xu(ju(zs(e),n[0]),n[1]-1);if(t===1/0)return xu(ju(zs(e),n[1]),n[0]);if(t===-1/0)return Cu(ju(zs(e),n[1]),n[0]);if(t===`fro`||t===`euclidean`)return Du(ju(ku(e),n));throw Error(`Error in norm: invalid ord value: ${t}`)}throw Error(`Error in norm: invalid axis: ${n}`)}var Pu=I({norm_:Mu});function Fu(e,t=null,n=!1){return Pu(e,`euclidean`,t,n)}var Iu=I({euclideanNorm_:Fu});function Lu(e){let t={x:F(e,`x`,`exp`)};return P.runKernel(`Exp`,t)}var Ru=I({exp_:Lu});function zu(e,t=0){let n=F(e,`x`,`expandDims`,`string_or_numeric`);f(t<=n.rank,()=>`Axis must be <= rank of the tensor`);let r={input:n},i={dim:t};return P.runKernel(bt,r,i)}var Bu=I({expandDims_:zu});function Vu(e){let t={x:F(e,`x`,`expm1`)};return P.runKernel(xt,t)}var Hu=I({expm1_:Vu});function Uu(e,t){let n=F(e,`x`,`tile`,`string_or_numeric`);f(n.rank===t.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${t}.`);let r={x:n},i={reps:t};return P.runKernel(er,r,i)}var Wu=I({tile_:Uu});function Gu(e,t,n,r=`float32`){t??=e;let i=L([e,t],r),a=e<=t?e:t;for(let e=0;e<a;++e)i.set(1,e,e);let o=V(i.toTensor(),[e,t]);if(n==null)return o;if(n.length===1)return Wu(Bu(o,0),[n[0],1,1]);if(n.length===2)return Wu(Bu(Bu(o,0),0),[n[0],n[1],1,1]);if(n.length===3)return Wu(Bu(Bu(Bu(o,0),0),0),[n[0],n[1],n[2],1,1]);throw Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}var Ku=I({eye_:Gu});function qu(e){let t={x:F(e,`x`,`floor`,`float32`)};return P.runKernel(wt,t)}var Ju=I({floor_:qu});function Yu(e,t,n=0,r=0){let i={x:F(e,`x`,`gather`),indices:F(t,`indices`,`gather`,`int32`)},a={axis:n,batchDims:r};return P.runKernel(Dt,i,a)}var Xu=I({gather_:Yu});function Zu(e,t){let n=F(e,`a`,`greater`,`string_or_numeric`),r=F(t,`b`,`greater`,`string_or_numeric`);[n,r]=Ci(n,r),z(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(kt,i)}var Qu=I({greater_:Zu});function $u(e,t){let n=F(e,`a`,`greaterEqual`,`string_or_numeric`),r=F(t,`b`,`greaterEqual`,`string_or_numeric`);[n,r]=Ci(n,r),z(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(At,i)}var ed=I({greaterEqual_:$u});function td(e){let t={x:F(e,`x`,`isFinite`)};return P.runKernel(Pt,t)}var nd=I({isFinite_:td});function rd(e){let t={x:F(e,`x`,`isInf`)};return P.runKernel(Ft,t)}var id=I({isInf_:rd});function ad(e){let t={x:F(e,`x`,`isNaN`)};return P.runKernel(It,t)}var od=I({isNaN_:ad});function sd(e,t=.2){let n={x:F(e,`x`,`leakyRelu`)},r={alpha:t};return P.runKernel(Lt,n,r)}var cd=I({leakyRelu_:sd});function ld(e,t){let n=F(e,`a`,`less`,`string_or_numeric`),r=F(t,`b`,`less`,`string_or_numeric`);[n,r]=Ci(n,r),z(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(Rt,i)}var ud=I({less_:ld});function dd(e,t){let n=F(e,`a`,`lessEqual`,`string_or_numeric`),r=F(t,`b`,`lessEqual`,`string_or_numeric`);[n,r]=Ci(n,r),z(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(zt,i)}var fd=I({lessEqual_:dd});function pd(e,t,n){if(n<=0)throw Error(`The number of values should be positive.`);let r={start:e,stop:t,num:n};return P.runKernel(Bt,{},r)}function md(e,t=5,n=1,r=1,i=.5){let a=F(e,`x`,`localResponseNormalization`);f(a.rank===4||a.rank===3,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${a.rank}.`),f(v(t),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${t}.`);let o=a,s=!1;a.rank===3&&(s=!0,o=V(a,[1,a.shape[0],a.shape[1],a.shape[2]]));let c={x:o},l={depthRadius:t,bias:n,alpha:r,beta:i},u=P.runKernel(`LRN`,c,l);return s?V(u,[u.shape[1],u.shape[2],u.shape[3]]):u}var hd=I({localResponseNormalization_:md});function gd(e){let t={x:F(e,`x`,`log`,`float32`)};return P.runKernel(`Log`,t)}var _d=I({log_:gd});function vd(e){let t={x:F(e,`x`,`log1p`)};return P.runKernel(Vt,t)}var yd=I({log1p_:vd});function bd(e){return P.customGrad(e)}function xd(e){let t={x:F(e,`x`,`softplus`)};return P.runKernel(Ln,t)}var Sd=I({softplus_:xd});function Cd(e){let t=F(e,`x`,`logSigmoid`);return bd(e=>({value:Qo(Sd(Qo(e))),gradFunc:t=>B(t,Ic(Qo(e)))}))(t)}var wd=I({logSigmoid_:Cd});function Td(e,t){let n=F(e,`a`,`sub`),r=F(t,`b`,`sub`);[n,r]=Ci(n,r);let i={a:n,b:r};return P.runKernel(`Sub`,i)}var W=I({sub_:Td});function Ed(e,t=-1){let n=F(e,`logits`,`logSoftmax`);if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${t}`);return bd((e,n)=>{let r=W(e,xu(e,t,!0)),i=W(io(r,`float32`),_d(ju(Ru(r),t,!0)));return n([i]),{value:i,gradFunc:(e,n)=>{let[r]=n,i=Ru(r);return W(e,B(ju(e,t,!0),i))}}})(n)}var Dd=I({logSoftmax_:Ed});function Od(e,t=null,n=!1){let r=F(e,`x`,`logSumExp`),i=C(t,r.shape),a=xu(r,i,!0),o=_d(ju(Ru(W(r,a)),i)),s=Ms(V(a,o.shape),o);return n?V(s,hu(s.shape,i)):s}var kd=I({logSumExp_:Od});function Ad(e,t){let n=F(e,`a`,`logicalAnd`,`bool`),r=F(t,`b`,`logicalAnd`,`bool`);z(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(Ht,i)}var jd=I({logicalAnd_:Ad});function Md(e){let t={x:F(e,`x`,`logicalNot`,`bool`)};return P.runKernel(Ut,t)}var Nd=I({logicalNot_:Md});function Pd(e,t){let n=F(e,`a`,`logicalOr`,`bool`),r=F(t,`b`,`logicalOr`,`bool`);z(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(Wt,i)}var Fd=I({logicalOr_:Pd});function Id(e,t){let n=F(e,`a`,`logicalXor`,`bool`),r=F(t,`b`,`logicalXor`,`bool`);return z(n.shape,r.shape),jd(Fd(e,t),Nd(jd(e,t)))}var Ld=I({logicalXor_:Id}),Rd=2147483648;function zd(e,t,n=`left`){let r=F(e,`sortedSequence`,`searchSorted`),i=F(t,`values`,`searchSorted`),a=r.shape[r.shape.length-1],o=i.shape[i.shape.length-1],s=V(r,[-1,a]),c=V(i,[-1,o]);if(s.rank<2)throw Error(`Sorted input argument must be at least 2-dimensional`);if(s.shape[0]!==c.shape[0])throw Error(`Leading dimension of 'sortedSequence' and 'values' must match.`);if(g(c.shape)>=Rd)throw Error(`values tensor size must less than ${Rd}`);if(s.shape[1]>=Rd)throw Error(`trailing dim_size must less than ${Rd} for int32 output type, was ${s.shape[1]}`);let l={sortedSequence:s,values:c},u={side:n};return P.runKernel(An,l,u)}var Bd=I({searchSorted_:zd});function Vd(e,t){return Bd(e,t,`left`)}function Hd(e,t,n,r,i){let a=F(e,`x`,`maxPool`),o=a,s=!1;a.rank===3&&(s=!0,o=V(a,[1,a.shape[0],a.shape[1],a.shape[2]])),f(o.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${o.rank}.`),f(Tc(n,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`),Dc(`maxPool`,r,i);let c={x:o},l={filterSize:t,strides:n,pad:r,dimRoundingMode:i},u=P.runKernel(qt,c,l);return s?V(u,[u.shape[1],u.shape[2],u.shape[3]]):u}var Ud=I({maxPool_:Hd});function Wd(e,t=[1,1,1],n,r,i,a=`NDHWC`){let o=F(e,`x`,`maxPool3d`),s=o,c=!1;o.rank===4&&(c=!0,s=V(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),f(s.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${s.rank}.`),f(a===`NDHWC`,()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),Dc(`maxPool3d`,r,i);let l={x:s},u={filterSize:t,strides:n,pad:r,dimRoundingMode:i,dataFormat:a},d=P.runKernel(Yt,l,u);return c?V(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}var Gd=I({maxPool3d_:Wd});function Kd(e,t,n,r,i=!1){let a={x:F(e,`x`,`maxPoolWithArgmax`)},o={filterSize:t,strides:n,pad:r,includeBatchInIndex:i},s=P.runKernel(Zt,a,o);return{result:s[0],indexes:s[1]}}var qd=I({maxPoolWithArgmax_:Kd});function Jd(e,t){let n=F(e,`a`,`maximum`),r=F(t,`b`,`maximum`);[n,r]=Ci(n,r),n.dtype===`bool`&&(n=io(n,`int32`),r=io(r,`int32`)),z(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(Kt,i)}var Yd=I({maximum_:Jd});function Xd(e,t=null,n=!1){let r={x:F(e,`x`,`mean`)},i={axis:t,keepDims:n};return P.runKernel(Qt,r,i)}var Zd=I({mean_:Xd});function Qd(e,t=`float32`){if(t===`complex64`)return Wi(Qd(e,`float32`),Qd(e,`float32`));let n=M(g(e),t);return P.makeTensor(n,e,t)}function $d(e,t=`float32`){if(t===`complex64`)return Wi($d(e,`float32`),Qd(e,`float32`));let n=ue(g(e),t);return P.makeTensor(n,e,t)}function ef(e,t,{indexing:n=`xy`}={}){if(n!==`xy`&&n!==`ij`)throw TypeError(`${n} is not a valid third argument to meshgrid`);if(e===void 0)return[];let r=F(e,`x`,`meshgrid`,e instanceof fi?e.dtype:`float32`);if(t===void 0)return[r];let i=F(t,`y`,`meshgrid`,t instanceof fi?t.dtype:`float32`),a=g(r.shape),o=g(i.shape);return n===`xy`?(r=V(r,[1,-1]),i=V(i,[-1,1]),[R($d([o,1],r.dtype),r),R(i,$d([1,a],i.dtype))]):(r=V(r,[-1,1]),i=V(i,[1,-1]),[R(r,$d([1,o],r.dtype)),R($d([a,1],i.dtype),i)])}function tf(e,t){let n=F(e,`a`,`minimum`),r=F(t,`b`,`minimum`);[n,r]=Ci(n,r),n.dtype===`bool`&&(n=io(n,`int32`),r=io(r,`int32`)),z(n.shape,r.shape);let i={a:n,b:r};return P.runKernel($t,i)}var nf=I({minimum_:tf});function rf(e,t,n){f(n===`reflect`||n===`symmetric`,()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${n}.`);let r=F(e,`x`,`mirrorPad`);if(r.rank===0)throw Error(`mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad`);f(t.length===r.rank,()=>`Padding doesn't match input. Must be ${r.rank}. Got ${t.length}.`);let i=+(n===`reflect`);for(let e=0;e<r.rank;e++)f(t[e].length===2,()=>`Invalid number of paddings. Must be length of 2 each.`),f(t[e][0]>=0&&t[e][0]<=r.shape[e]-i&&t[e][1]>=0&&t[e][1]<=r.shape[e]-i,()=>`Padding in dimension ${e} cannot be greater than or equal to ${r.shape[e]-i} or less than 0 for input of shape ${r.shape}`);let a={paddings:t,mode:n},o={x:r};return P.runKernel(en,o,a)}var af=I({mirrorPad_:rf});function of(e,t){let n=F(e,`a`,`mod`),r=F(t,`b`,`mod`);[n,r]=Ci(n,r);let i={a:n,b:r};return P.runKernel(`Mod`,i)}var sf=I({mod_:of});function cf(e,t=null,n=!1){e=F(e,`x`,`moments`);let r=C(t,e.shape),i=Zd(e,r,n),a=i.shape;return n||(a=hu(i.shape,r)),{mean:i,variance:Zd(ku(W(io(e,`float32`),V(i,a))),r,n)}}var lf=I({moments_:cf});function uf(e,t,n,r){let i=F(t,`data`,`multiRNNCell`),a=Vi(n,`c`,`multiRNNCell`),o=Vi(r,`h`,`multiRNNCell`),s=i,c=[];for(let t=0;t<e.length;t++){let n=e[t](s,a[t],o[t]);c.push(n[0]),c.push(n[1]),s=n[1]}let l=[],u=[];for(let e=0;e<c.length;e+=2)l.push(c[e]),u.push(c[e+1]);return[l,u]}var df=I({multiRNNCell_:uf});function ff(e,t,n,r=!1){let i=F(e,`logits`,`multinomial`),a=i.size,o=i.rank;if(a<2)throw Error(`Error in multinomial: you need at least 2 outcomes, but got ${a}.`);if(o>2)throw Error(`Rank of probabilities must be 1 or 2, but is ${o}`);n||=Math.random();let s={logits:o===1?V(i,[1,-1]):i},c={numSamples:t,seed:n,normalized:r},l=P.runKernel(tn,s,c);return o===1?V(l,[l.size]):l}var pf=I({multinomial_:ff});function mf(e,t){let n=F(e,`a`,`notEqual`,`string_or_numeric`),r=F(t,`b`,`notEqual`,`string_or_numeric`);[n,r]=Ci(n,r),z(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(rn,i)}var hf=I({notEqual_:mf});function gf(e){let t={x:F(e,`x`,`onesLike`)};return P.runKernel(cn,t)}var _f=I({onesLike_:gf});function vf(e,t){let n=F(e,`v1`,`outerProduct`),r=F(t,`v2`,`outerProduct`);return f(n.rank===1&&r.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${n.rank} and ${r.rank}.`),R(V(n,[-1,1]),V(r,[1,-1]))}var yf=I({outerProduct_:vf});function bf(e,t,n=0){let r=F(e,`x`,`pad`);if(r.rank===0)throw Error(`pad(scalar) is not defined. Pass non-scalar to pad`);let i={paddings:t,constantValue:n},a={x:r};return P.runKernel(dn,a,i)}var xf=I({pad_:bf});function Sf(e,t,n=0){return f(t.length===2,()=>`Invalid number of paddings. Must be length of 2.`),xf(e,[t],n)}var Cf=I({pad1d_:Sf});function wf(e,t,n=0){return f(t.length===2&&t[0].length===2&&t[1].length===2,()=>`Invalid number of paddings. Must be length of 2 each.`),xf(e,t,n)}var Tf=I({pad2d_:wf});function Ef(e,t,n=0){return f(t.length===3&&t[0].length===2&&t[1].length===2&&t[2].length===2,()=>`Invalid number of paddings. Must be length of 2 each.`),xf(e,t,n)}var Df=I({pad3d_:Ef});function Of(e,t,n=0){return f(t.length===4&&t[0].length===2&&t[1].length===2&&t[2].length===2&&t[3].length===2,()=>`Invalid number of paddings. Must be length of 2 each.`),xf(e,t,n)}var kf=I({pad4d_:Of});function Af(e,t,n){let r=F(e,`x`,`spaceToBatchND`);f(r.rank>=1+t.length,()=>`input rank ${r.rank} should be > than [blockShape] ${t.length}`),f(n.length===t.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${t.length}`),f(r.shape.reduce((e,r,i)=>i>0&&i<=t.length?e&&(r+n[i-1][0]+n[i-1][1])%t[i-1]===0:e,!0),()=>`input spatial dimensions ${r.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${t.toString()}`);let i={x:r},a={blockShape:t,paddings:n};return P.runKernel(zn,i,a)}var jf=I({spaceToBatchND_:Af});function Mf(e,t,n,r,i,a,o){i??=[1,1],a??=1,r===0&&(r=`valid`);let s=F(e,`x`,`maxPool`),c=s,l=!1;s.rank===3&&(l=!0,c=V(s,[1,s.shape[0],s.shape[1],s.shape[2]])),f(Tc(a,i),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${a} and dilations '${i}'`);let u=dc(c.shape,t,a,i,r),d=[u.dilationHeight,u.dilationWidth],p;p=r===`same`?Pf([u.filterHeight,u.filterWidth],d):[[0,0],[0,0]];let m=d[0]===1&&d[1]===1,[h,g]=Nf([u.inHeight,u.inWidth],d,p),_=m?r:`valid`,v=m?c:jf(c,d,h),y=(n===`avg`?()=>Ac(v,t,a,_,o):()=>Ud(v,t,a,_,o))(),b=m?y:Uc(y,d,g);return l?V(b,[b.shape[1],b.shape[2],b.shape[3]]):b}function Nf(e,t,n){let r=n.map(e=>e[0]),i=n.map(e=>e[1]),a=e.concat(r,i),o=t.map((e,t)=>(e-a[t]%e)%e),s=i.map((e,t)=>e+o[t]);return[t.map((e,t)=>[r[t],s[t]]),t.map((e,t)=>[0,o[t]])]}function Pf(e,t){let n=e.map((e,n)=>e+(e-1)*(t[n]-1)).map(e=>e-1),r=n.map(e=>Math.floor(e/2)),i=n.map((e,t)=>e-r[t]);return n.map((e,t)=>[r[t],i[t]])}var Ff=I({pool_:Mf});function If(e,t){let n={x:F(e,`x`,`prelu`),alpha:F(t,`alpha`,`prelu`)};return P.runKernel(fn,n)}var Lf=I({prelu_:If});function Rf(e,t=null,n=!1){let r=F(e,`x`,`prod`);r.dtype===`bool`&&(r=io(r,`int32`));let i={x:r},a={axis:t,keepDims:n};return P.runKernel(pn,i,a)}var zf=I({prod_:Rf});function Bf(e,t,n,r){let i={paramsNestedSplits:e.map((e,t)=>F(e,`tensors${t}`,`raggedGather`,`int32`)),paramsDenseValues:F(t,`paramsDenseValues`,`raggedGather`),indices:F(n,`indices`,`raggedGather`,`int32`)},a={outputRaggedRank:r},o=P.runKernel(mn,i,a);return{outputNestedSplits:o.slice(0,o.length-1),outputDenseValues:o[o.length-1]}}var Vf=I({raggedGather_:Bf});function Hf(e,t,n,r,i){let a=F(e,`shape`,`raggedTensorToTensor`,`int32`),o=F(t,`values`,`raggedTensorToTensor`),s={shape:a,values:o,defaultValue:F(n,`defaultValue`,`raggedTensorToTensor`,o.dtype),rowPartitionTensors:r.map((e,t)=>F(e,`tensors${t}`,`raggedTensorToTensor`,`int32`))},c={rowPartitionTypes:i};return P.runKernel(hn,s,c)}var Uf=I({raggedTensorToTensor_:Hf});function Wf(e,t,n){let r=g(e),i=null;if(n==null||n===`float32`)i=new Float32Array(r);else if(n===`int32`)i=new Int32Array(r);else if(n===`bool`)i=new Uint8Array(r);else throw Error(`Unknown data type ${n}`);for(let e=0;e<r;e++)i[e]=t();return P.makeTensor(i,e,n)}var Gf=I({rand_:Wf}),Kf=e(((e,t,n)=>{(function(e,t,n){function r(e){var t=this,n=o();t.next=function(){var e=2091639*t.s0+t.c*23283064365386963e-26;return t.s0=t.s1,t.s1=t.s2,t.s2=e-(t.c=e|0)},t.c=1,t.s0=n(` `),t.s1=n(` `),t.s2=n(` `),t.s0-=n(e),t.s0<0&&(t.s0+=1),t.s1-=n(e),t.s1<0&&(t.s1+=1),t.s2-=n(e),t.s2<0&&(t.s2+=1),n=null}function i(e,t){return t.c=e.c,t.s0=e.s0,t.s1=e.s1,t.s2=e.s2,t}function a(e,t){var n=new r(e),a=t&&t.state,o=n.next;return o.int32=function(){return n.next()*4294967296|0},o.double=function(){return o()+(o()*2097152|0)*11102230246251565e-32},o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}function o(){var e=4022871197;return function(t){t=String(t);for(var n=0;n<t.length;n++){e+=t.charCodeAt(n);var r=.02519603282416938*e;e=r>>>0,r-=e,r*=e,e=r>>>0,r-=e,e+=r*4294967296}return(e>>>0)*23283064365386963e-26}}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.alea=a})(n,typeof t==`object`&&t,typeof define==`function`&&define)})),qf=e(((e,t,n)=>{(function(e,t,n){function r(e){var t=this,n=``;t.x=0,t.y=0,t.z=0,t.w=0,t.next=function(){var e=t.x^t.x<<11;return t.x=t.y,t.y=t.z,t.z=t.w,t.w^=t.w>>>19^e^e>>>8},e===(e|0)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=n.charCodeAt(r)|0,t.next()}function i(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xor128=a})(n,typeof t==`object`&&t,typeof define==`function`&&define)})),Jf=e(((e,t,n)=>{(function(e,t,n){function r(e){var t=this,n=``;t.next=function(){var e=t.x^t.x>>>2;return t.x=t.y,t.y=t.z,t.z=t.w,t.w=t.v,(t.d=t.d+362437|0)+(t.v=t.v^t.v<<4^(e^e<<1))|0},t.x=0,t.y=0,t.z=0,t.w=0,t.v=0,e===(e|0)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=n.charCodeAt(r)|0,r==n.length&&(t.d=t.x<<10^t.x>>>4),t.next()}function i(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t.v=e.v,t.d=e.d,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xorwow=a})(n,typeof t==`object`&&t,typeof define==`function`&&define)})),Yf=e(((e,t,n)=>{(function(e,t,n){function r(e){var t=this;t.next=function(){var e=t.x,n=t.i,r=e[n],i;return r^=r>>>7,i=r^r<<24,r=e[n+1&7],i^=r^r>>>10,r=e[n+3&7],i^=r^r>>>3,r=e[n+4&7],i^=r^r<<7,r=e[n+7&7],r^=r<<13,i^=r^r<<9,e[n]=i,t.i=n+1&7,i};function n(e,t){var n,r=[];if(t===(t|0))r[0]=t;else for(t=``+t,n=0;n<t.length;++n)r[n&7]=r[n&7]<<15^t.charCodeAt(n)+r[n+1&7]<<13;for(;r.length<8;)r.push(0);for(n=0;n<8&&r[n]===0;++n);for(n==8?r[7]=-1:r[n],e.x=r,e.i=0,n=256;n>0;--n)e.next()}n(t,e)}function i(e,t){return t.x=e.x.slice(),t.i=e.i,t}function a(e,t){e??=+new Date;var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(a.x&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xorshift7=a})(n,typeof t==`object`&&t,typeof define==`function`&&define)})),Xf=e(((e,t,n)=>{(function(e,t,n){function r(e){var t=this;t.next=function(){var e=t.w,n=t.X,r=t.i,i,a;return t.w=e=e+1640531527|0,a=n[r+34&127],i=n[r=r+1&127],a^=a<<13,i^=i<<17,a^=a>>>15,i^=i>>>12,a=n[r]=a^i,t.i=r,a+(e^e>>>16)|0};function n(e,t){var n,r,i,a,o,s=[],c=128;for(t===(t|0)?(r=t,t=null):(t+=`\0`,r=0,c=Math.max(c,t.length)),i=0,a=-32;a<c;++a)t&&(r^=t.charCodeAt((a+32)%t.length)),a===0&&(o=r),r^=r<<10,r^=r>>>15,r^=r<<4,r^=r>>>13,a>=0&&(o=o+1640531527|0,n=s[a&127]^=r+o,i=n==0?i+1:0);for(i>=128&&(s[(t&&t.length||0)&127]=-1),i=127,a=512;a>0;--a)r=s[i+34&127],n=s[i=i+1&127],r^=r<<13,n^=n<<17,r^=r>>>15,n^=n>>>12,s[i]=r^n;e.w=o,e.X=s,e.i=i}n(t,e)}function i(e,t){return t.i=e.i,t.w=e.w,t.X=e.X.slice(),t}function a(e,t){e??=+new Date;var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(a.X&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xor4096=a})(n,typeof t==`object`&&t,typeof define==`function`&&define)})),Zf=e(((e,t,n)=>{(function(e,t,n){function r(e){var t=this,n=``;t.next=function(){var e=t.b,n=t.c,r=t.d,i=t.a;return e=e<<25^e>>>7^n,n=n-r|0,r=r<<24^r>>>8^i,i=i-e|0,t.b=e=e<<20^e>>>12^n,t.c=n=n-r|0,t.d=r<<16^n>>>16^i,t.a=i-e|0},t.a=0,t.b=0,t.c=-1640531527,t.d=1367130551,e===Math.floor(e)?(t.a=e/4294967296|0,t.b=e|0):n+=e;for(var r=0;r<n.length+20;r++)t.b^=n.charCodeAt(r)|0,t.next()}function i(e,t){return t.a=e.a,t.b=e.b,t.c=e.c,t.d=e.d,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.tychei=a})(n,typeof t==`object`&&t,typeof define==`function`&&define)})),Qf=e(((e,t,n)=>{(function(e,n,r){var i=256,a=6,o=52,s=`random`,c=r.pow(i,a),l=r.pow(2,o),u=l*2,d=i-1,f;function p(e,t,o){var d=[];t=t==1?{entropy:!0}:t||{};var f=_(g(t.entropy?[e,y(n)]:e??v(),3),d),p=new m(d),b=function(){for(var e=p.g(a),t=c,n=0;e<l;)e=(e+n)*i,t*=i,n=p.g(1);for(;e>=u;)e/=2,t/=2,n>>>=1;return(e+n)/t};return b.int32=function(){return p.g(4)|0},b.quick=function(){return p.g(4)/4294967296},b.double=b,_(y(p.S),n),(t.pass||o||function(e,t,n,i){return i&&(i.S&&h(i,p),e.state=function(){return h(p,{})}),n?(r[s]=e,t):e})(b,f,`global`in t?t.global:this==r,t.state)}function m(e){var t,n=e.length,r=this,a=0,o=r.i=r.j=0,s=r.S=[];for(n||(e=[n++]);a<i;)s[a]=a++;for(a=0;a<i;a++)s[a]=s[o=d&o+e[a%n]+(t=s[a])],s[o]=t;(r.g=function(e){for(var t,n=0,a=r.i,o=r.j,s=r.S;e--;)t=s[a=d&a+1],n=n*i+s[d&(s[a]=s[o=d&o+t])+(s[o]=t)];return r.i=a,r.j=o,n})(i)}function h(e,t){return t.i=e.i,t.j=e.j,t.S=e.S.slice(),t}function g(e,t){var n=[],r=typeof e,i;if(t&&r==`object`)for(i in e)try{n.push(g(e[i],t-1))}catch{}return n.length?n:r==`string`?e:e+`\0`}function _(e,t){for(var n=e+``,r,i=0;i<n.length;)t[d&i]=d&(r^=t[d&i]*19)+n.charCodeAt(i++);return y(t)}function v(){try{var t;return f&&(t=f.randomBytes)?t=t(i):(t=new Uint8Array(i),(e.crypto||e.msCrypto).getRandomValues(t)),y(t)}catch{var r=e.navigator,a=r&&r.plugins;return[+new Date,e,a,e.screen,y(n)]}}function y(e){return String.fromCharCode.apply(0,e)}if(_(r.random(),n),typeof t==`object`&&t.exports){t.exports=p;try{f=$a()}catch{}}else typeof define==`function`&&define.amd?define(function(){return p}):r[`seed`+s]=p})(typeof self<`u`?self:n,[],Math)})),$f=n(e(((e,t)=>{var n=Kf(),r=qf(),i=Jf(),a=Yf(),o=Xf(),s=Zf(),c=Qf();c.alea=n,c.xor128=r,c.xorwow=i,c.xorshift7=a,c.xor4096=o,c.tychei=s,t.exports=c}))()),ep=class{constructor(e,t,n,r,i){this.mean=e,this.stdDev=t,this.dtype=n,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);let a=i||Math.random();this.random=$f.alea(a.toString())}nextValue(){if(!isNaN(this.nextVal)){let e=this.nextVal;return this.nextVal=NaN,e}let e,t,n=!1;for(;!n;){let r,i,a;do r=2*this.random()-1,i=2*this.random()-1,a=r*r+i*i;while(a>=1||a===0);let o=Math.sqrt(-2*Math.log(a)/a);e=this.mean+this.stdDev*r*o,t=this.mean+this.stdDev*i*o,(!this.truncated||this.isValidTruncated(e))&&(n=!0)}return(!this.truncated||this.isValidTruncated(t))&&(this.nextVal=this.convertValue(t)),this.convertValue(e)}convertValue(e){return this.dtype==null||this.dtype===`float32`?e:Math.round(e)}isValidTruncated(e){return e<=this.upper&&e>=this.lower}},tp=class{constructor(e,t,n,r){this.alpha=e,this.beta=1/t,this.dtype=n;let i=r||Math.random();this.randu=$f.alea(i.toString()),this.randn=new ep(0,1,n,!1,this.randu()),this.d=e<1?e+2/3:e-1/3,this.c=1/Math.sqrt(9*this.d)}nextValue(){let e,t,n,r,i,a;for(;;){do r=this.randn.nextValue(),a=1+this.c*r;while(a<=0);if(a*=a*a,e=r*r,t=1-.331*e*e,n=.5*e+this.d*(1-a+Math.log(a)),i=this.randu(),i<t||Math.log(i)<n)break}return a=1/this.beta*this.d*a,this.alpha<1&&(a*=this.randu()**(1/this.alpha)),this.convertValue(a)}convertValue(e){return this.dtype===`float32`?e:Math.round(e)}},np=class{constructor(e=0,t=1,n,r){if(this.canReturnFloat=()=>this.dtype==null||this.dtype===`float32`,this.min=e,this.range=t-e,this.dtype=n,r??=Math.random(),typeof r==`number`&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw Error(`The difference between ${e} - ${t} <= 1 and dtype is not float`);this.random=$f.alea(r)}convertValue(e){return this.canReturnFloat()?e:Math.round(e)}nextValue(){return this.convertValue(this.min+this.range*this.random())}};function rp(e,t,n=1,r=`float32`,i){if(n??=1,r??=`float32`,r!==`float32`&&r!==`int32`)throw Error(`Unsupported data type ${r}`);let a=new tp(t,n,r,i),o=L(e,r);for(let e=0;e<o.values.length;e++)o.values[e]=a.nextValue();return o.toTensor()}var ip=I({randomGamma_:rp});function ap(e,t=0,n=1,r,i){if(r!=null&&r===`bool`)throw Error(`Unsupported data type ${r}`);let a=new ep(t,n,r,!1,i),o=L(e,r);for(let e=0;e<o.values.length;e++)o.values[e]=a.nextValue();return o.toTensor()}var op=I({randomNormal_:ap});function sp(e,t,n){if(t!=null&&t===`bool`)throw Error(`Unsupported data type ${t}`);return op(e,0,1,t,n)}var cp=I({randomStandardNormal_:sp});function lp(e,t=0,n=1,r=`float32`,i){let a=L(e,r),o=new np(t,n,null,i);for(let e=0;e<a.values.length;e++)a.values[e]=o.nextValue();return a.toTensor()}var up=I({randomUniform_:lp});function dp(e,t,n=1,r=`float32`){if(n===0)throw Error(`Cannot have a step of zero`);let i={start:e,stop:t,step:n,dtype:r};return P.runKernel(gn,{},i)}function fp(e){let t={x:F(e,`x`,`reciprocal`)};return P.runKernel(vn,t)}var pp=I({reciprocal_:fp});function mp(e){let t={x:F(e,`x`,`relu`)};return P.runKernel(yn,t)}var hp=I({relu_:mp});function gp(e){let t={x:F(e,`x`,`relu6`)};return P.runKernel(Tn,t)}var _p=I({relu6_:gp});function vp(e,t){let n={x:F(e,`x`,`reverse`)},r={dims:t};return P.runKernel(En,n,r)}var yp=I({reverse_:vp});function bp(e){let t=F(e,`x`,`reverse`);return f(t.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${t.rank}.`),yp(t,0)}var xp=I({reverse1d_:bp});function Sp(e,t){let n=F(e,`x`,`reverse`);return f(n.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${n.rank}.`),yp(n,t)}var Cp=I({reverse2d_:Sp});function wp(e,t){let n=F(e,`x`,`reverse`);return f(n.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${n.rank}.`),yp(n,t)}var Tp=I({reverse3d_:wp});function Ep(e,t){let n=F(e,`x`,`reverse`);return f(n.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${n.rank}.`),yp(n,t)}var Dp=I({reverse4d_:Ep});function Op(e){let t={x:F(e,`x`,`round`)};return P.runKernel(Dn,t)}var kp=I({round_:Op});function Ap(e){let t={x:F(e,`x`,`rsqrt`,`float32`)};return P.runKernel(On,t)}var jp=I({rsqrt_:Ap});function Mp(e){let t={x:F(e,`x`,`selu`)};return P.runKernel(Mn,t)}var Np=I({selu_:Mp});function Pp(e,t,n,r,i,a=[1,1],o=`NHWC`){let s=F(e,`x`,`separableConv2d`),c=F(t,`depthwiseFilter`,`separableConv2d`),l=F(n,`pointwiseFilter`,`separableConv2d`),u=s,d=!1;if(s.rank===3&&(d=!0,u=V(s,[1,s.shape[0],s.shape[1],s.shape[2]])),o===`NCHW`)throw Error(`separableConv2d currently does not support dataFormat NCHW; only NHWC is supported`);f(u.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${u.rank}.`),f(c.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${c.rank}.`),f(l.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${c.rank}.`),f(l.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${l.shape[0]}.`),f(l.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${l.shape[1]}.`);let p=c.shape[2],m=c.shape[3];f(l.shape[2]===p*m,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${p*m}, but got ${l.shape[2]}.`);let h=yl(Gl(u,c,r,i,o,a),l,1,`valid`,o);return d?V(h,[h.shape[1],h.shape[2],h.shape[3]]):h}var Fp=I({separableConv2d_:Pp});async function Ip(e,t){let n=F(e,`x`,`setdiff1d`),r=F(t,`y`,`setdiff1d`);f(n.dtype===r.dtype,()=>`x and y should have the same dtype, but got x (${n.dtype}) and y (${r.dtype}).`),f(n.rank===1,()=>`x should be 1D tensor, but got x (${n.shape}).`),f(r.rank===1,()=>`y should be 1D tensor, but got y (${r.shape}).`);let i=await n.data(),a=await r.data(),o=new Set(a),s=0;for(let e=0;e<i.length;e++)o.has(i[e])||s++;let c=new si([s],n.dtype),l=new si([s],`int32`);for(let e=0,t=0;e<i.length;e++)o.has(i[e])||(c.values[t]=i[e],l.values[t]=e,t++);return[c.toTensor(),l.toTensor()]}var Lp=Ip;function Rp(e){let t={x:F(e,`x`,`sign`)};return P.runKernel(Fn,t)}var zp=I({sign_:Rp});function Bp(e){let t={x:F(e,`x`,`sin`,`float32`)};return P.runKernel(`Sin`,t)}var Vp=I({sin_:Bp});function Hp(e){let t={x:F(e,`x`,`sinh`)};return P.runKernel(Pn,t)}var Up=I({sinh_:Hp});function Wp(e,t,n){let r=F(e,`x`,`slice1d`);return f(r.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${r.rank} tensor`),H(r,[t],[n])}var Gp=I({slice1d_:Wp});function Kp(e,t,n){let r=F(e,`x`,`slice2d`);return f(r.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${r.rank} tensor`),H(r,t,n)}var qp=I({slice2d_:Kp});function Jp(e,t,n){let r=F(e,`x`,`slice3d`);return f(r.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${r.rank} tensor`),H(r,t,n)}var Yp=I({slice3d_:Jp});function Xp(e,t,n){let r=F(e,`x`,`slice4d`);return f(r.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${r.rank} tensor`),H(r,t,n)}var Zp=I({slice4d_:Xp});function Qp(e,t=-1){let n=F(e,`logits`,`softmax`,`float32`);if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${t}`);let r={logits:n},i={dim:t};return P.runKernel(Vn,r,i)}var $p=I({softmax_:Qp});function em(e){f(e.dtype===`complex64`,()=>`The dtype for tf.spectral.fft() must be complex64 but got ${e.dtype}.`);let t={input:e};return P.runKernel(`FFT`,t)}var tm=I({fft_:em});function nm(e){f(e.dtype===`complex64`,()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${e.dtype}.`);let t={input:e};return P.runKernel(Mt,t)}var rm=I({ifft_:nm});function im(e){let t=e.shape[e.shape.length-1],n=e.size/t,r;if(t<=2)r=rm(V(e,[n,t]));else{let i=[n,2*(t-1)],a=V(es(e),[n,t]),o=V(Xo(e),[n,t]),s=yp(H(a,[0,1],[n,t-2]),1),c=B(yp(H(o,[0,1],[n,t-2]),1),U(-1));r=rm(V(Wi(Pc([a,s],1),Pc([o,c],1)),[i[0],i[1]]))}if(r=es(r),e.rank===3&&e.shape[0]!==0){let t=r,n=e.shape[0];r=V(r,[n,r.shape[0]/n,r.shape[1]]),t.dispose()}return r}var am=I({irfft_:im});function om(e,t,n=0){let r={x:F(e,`x`,`split`)},i={numOrSizeSplits:t,axis:n};return P.runKernel(Bn,r,i)}var sm=I({split_:om});function cm(e,t){f(e.dtype===`float32`,()=>`The dtype for rfft() must be real value but got ${e.dtype}`);let n=e.shape[e.shape.length-1],r=e.size/n,i;if(t!=null&&t<n){let r=e.shape.map(e=>0),a=e.shape.map(e=>e);a[e.shape.length-1]=t,i=H(e,r,a),n=t}else if(t!=null&&t>n){let r=e.shape.map(e=>e);r[e.shape.length-1]=t-n,i=Pc([e,Qd(r)],e.shape.length-1),n=t}else i=e;let a=tu(i),o=tm(V(Wi(i,a),[r,n])),s=Math.floor(n/2)+1,c=es(o),l=Xo(o),u=sm(c,[s,n-s],c.shape.length-1),d=sm(l,[s,n-s],l.shape.length-1),p=i.shape.slice();return p[i.shape.length-1]=s,V(Wi(u[0],d[0]),p)}var lm=I({rfft_:cm});function um(e,t){let n=F(e,`a`,`squaredDifference`),r=F(t,`b`,`squaredDifference`);[n,r]=Ci(n,r),z(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(qn,i,{})}var dm=I({squaredDifference_:um});function fm(e,t){let n=F(e,`x`,`squeeze`,`string_or_numeric`);return V(n,w(n.shape,t).newShape)}var pm=I({squeeze_:fm});function mm(e,t=0){let n=Vi(e,`tensors`,`stack`,`string_or_numeric`);f(n.length>=1,()=>`Pass at least one tensor to tf.stack`),n.length>0&&f(t<=n[0].rank,()=>`Axis must be <= rank of the tensor`);let r=n,i={axis:t};return P.runKernel(un,r,i)}var hm=I({stack_:mm});function gm(e,t=0){let n={x:F(e,`x`,`step`)},r={alpha:t};return P.runKernel(cr,n,r)}var _m=I({step_:gm});function vm(e,t,n,r,i=0,a=0,o=0,s=0,c=0){let l={x:F(e,`x`,`stridedSlice`,`string_or_numeric`)},u={begin:t,end:n,strides:r,beginMask:i,endMask:a,ellipsisMask:o,newAxisMask:s,shrinkAxisMask:c};return P.runKernel(Yn,l,u)}var ym=I({stridedSlice_:vm});function bm(e){let t={x:F(e,`x`,`tan`,`float32`)};return P.runKernel(`Tan`,t)}var xm=I({tan_:bm});function Sm(e,t){m(e);let n=Ri(e,t);if(n.length!==1)throw Error(`tensor1d() requires values to be a flat/TypedArray`);return Gi(e,null,n,t)}function Cm(e,t,n){if(m(e),t!=null&&t.length!==2)throw Error(`tensor2d() requires shape to have two numbers`);let r=Ri(e,n);if(r.length!==2&&r.length!==1)throw Error(`tensor2d() requires values to be number[][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return Gi(e,t,r,n)}function wm(e,t,n){if(m(e),t!=null&&t.length!==4)throw Error(`tensor4d() requires shape to have four numbers`);let r=Ri(e,n);if(r.length!==4&&r.length!==1)throw Error(`tensor4d() requires values to be number[][][][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor4d() requires shape to be provided when `values` are a flat array");return Gi(e,t,r,n)}function Tm(e,t,n){if(m(e),t!=null&&t.length!==5)throw Error(`tensor5d() requires shape to have five numbers`);let r=Ri(e,n);if(r.length!==5&&r.length!==1)throw Error(`tensor5d() requires values to be number[][][][][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor5d() requires shape to be provided when `values` are a flat array");return Gi(e,t,r,n)}function Em(e,t,n){if(m(e),t!=null&&t.length!==6)throw Error(`tensor6d() requires shape to have six numbers`);let r=Ri(e,n);if(r.length!==6&&r.length!==1)throw Error(`tensor6d() requires values to be number[][][][][][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor6d() requires shape to be provided when `values` are a flat array");return t||=r,Gi(e,t,r,n)}function Dm(e,t=1,n=!0){let r=F(e,`x`,`topk`);if(r.rank===0)throw Error(`topk() expects the input to be of rank 1 or higher`);let i=r.shape[r.shape.length-1];if(t<0)throw Error(`'k' passed to topk() must be >= 0 but got ${t}`);if(t>i)throw Error(`'k' passed to topk() must be <= the last dimension (${i}) but got ${t}`);let a={x:r},o={k:t,sorted:n},[s,c]=P.runKernel(tr,a,o);return{values:s,indices:c}}var Om=I({topk_:Dm});function km(e,t=0,n=1,r,i){if(r!=null&&r===`bool`)throw Error(`Unsupported data type $ { dtype }`);let a=new ep(t,n,r,!0,i),o=L(e,r);for(let e=0;e<o.values.length;e++)o.values[e]=a.nextValue();return o.toTensor()}var Am=I({truncatedNormal_:km});function jm(e,t=0){let n=F(e,`x`,`unique`,`string_or_numeric`);f(n.rank>0,()=>`The input tensor must be at least 1D`);let r={x:n},i={axis:t},[a,o]=P.runKernel(ir,r,i);return{values:a,indices:o}}var Mm=I({unique_:jm});function Nm(e,t,n){let r=F(e,`x`,`unsortedSegmentSum`),i=F(t,`segmentIds`,`unsortedSegmentSum`,`int32`);f(v(n),()=>`numSegments must be of dtype int`);let a={x:r,segmentIds:i},o={numSegments:n};return P.runKernel(or,a,o)}var Pm=I({unsortedSegmentSum_:Nm});function Fm(e,t=0){let n=F(e,`x`,`unstack`,`string_or_numeric`);f(t>=-n.shape.length&&t<n.shape.length,()=>`Axis = ${t} is not in [-${n.shape.length}, ${n.shape.length})`);let r={value:n},i={axis:t};return P.runKernel(ar,r,i)}var Im=I({unstack_:Fm});function Lm(e,t){return Bd(e,t,`right`)}function Rm(e,t=!0,n,r){return P.makeVariable(e,t,n,r)}function zm(e,t){let n=[];for(let e=0;e<t.length;e++)t[e]&&n.push(e);let r=L(e,`int32`),i=L([n.length,e.length],`int32`);for(let t=0;t<n.length;t++){let a=r.indexToLoc(n[t]),o=t*e.length;i.values.set(a,o)}return i.toTensor()}async function Bm(e){let t=F(e,`condition`,`whereAsync`,`bool`),n=await t.data(),r=zm(t.shape,n);return e!==t&&t.dispose(),r}var Vm=Bm;async function Hm(e,t,n){let r=F(e,`tensor`,`boolMask`),i=F(t,`mask`,`boolMask`,`bool`),a=n??0,o=i.rank,s=r.shape;f(o>0,()=>`mask cannot be scalar`),p(s.slice(a,a+o),i.shape,`mask's shape must match the first K dimensions of tensor's shape,`);let c=1;for(let e=a;e<a+o;e++)c*=s[e];let l=V(r,s.slice(0,a).concat([c],s.slice(a+o))),u=V(i,[-1]),d=await Vm(u),m=pm(d,[1]),h=Xu(l,m,a);return e!==r&&r.dispose(),t!==i&&i.dispose(),m.dispose(),l.dispose(),u.dispose(),d.dispose(),h}var Um=Hm;function Wm(e,t,n,r,i=!0){let a=F(e,`v`,`movingAverage`),o=F(t,`x`,`movingAverage`),s=F(n,`decay`,`movingAverage`);wi(a,o),f(_(a.shape,o.shape),()=>`Shape mismatch in v and x`);let c=U(1),l=W(c,s),u=B(W(o,a),l);if(i){f(r!=null,()=>`When using zeroDebias: true, step is required.`);let e=F(r,`step`,`movingAverage`);u=Is(u,W(c,Tu(s,e)))}return Ms(a,u)}var Gm=I({movingAverage_:Wm});function Km(e,t,n){let r=F(e,`indices`,`scatterND`,`int32`),i=F(t,`updates`,`scatterND`);cs(i,r,n);let a={indices:r,updates:i},o={shape:n};return P.runKernel(kn,a,o)}var qm=I({scatterND_:Km});function Jm(e,t,n,r){if(e.dtype!==`int32`)throw Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.rank>2)throw Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${e.shape}.`);let i=e.rank>0?e.shape[0]:1,a=e.rank>1?e.shape[1]:1;if(n.length!==a)throw Error(`outputShape has incorrect number of elements:, ${n.length}, should be: ${a}.`);let o=t.size;if(!(t.rank===0||t.rank===1&&o===i))throw Error(`sparseValues has incorrect shape ${t.shape}, should be [] or [${i}]`);if(t.dtype!==r.dtype)throw Error(`sparseValues.dtype must match defaultValues.dtype`)}function Ym(e,t,n,r=0){let i=F(e,`sparseIndices`,`sparseToDense`,`int32`),a=F(t,`sparseValues`,`sparseToDense`,`string_or_numeric`),o=F(r,`defaultValue`,`sparseToDense`,a.dtype);Jm(i,a,n,o);let s={sparseIndices:i,sparseValues:a,defaultValue:o},c={outputShape:n};return P.runKernel(Kn,s,c)}var Xm=I({sparseToDense_:Ym});function Zm(e,t){let n=F(t,`indices`,`gatherND`,`int32`),r={params:F(e,`x`,`gatherND`,`string_or_numeric`),indices:n};return P.runKernel(Ot,r)}var Qm=I({gatherND_:Zm});function $m(e,t){if(t==null)return e.shape.slice();if(_(e.shape,t))return t;if(e.shape.length===t.length){let n=[];for(let r=0;r<e.shape.length;r++)t[r]==null&&e.shape[r]!=null?n.push(e.shape[r]):n.push(t[r]);return n}return t}function eh(e,t,n,r){let i=F(e,`x`,`dropout`);if(f(i.dtype===`float32`,()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${i.dtype} tensor instead.`),f(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),t===0)return e instanceof fi?i.clone():i;let a=$m(i,n),o=1-t;return B(i,Is(Ju(Ms(up(a,0,1,`float32`,r),o)),o))}var th=I({dropout_:eh});function nh(e){return Math.floor(2**Math.ceil(Math.log(e)/Math.log(2)))}function rh(e,t,n){let r=1-e%2,i=new Float32Array(e);for(let a=0;a<e;++a){let o=2*Math.PI*a/(e+r-1);i[a]=t-n*Math.cos(o)}return Sm(i,`float32`)}async function ih(e,t,n=1){let r=F(e,`predictions`,`inTopK`),i=F(t,`targets`,`inTopK`);f(r.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${r.rank}`),f(r.rank-1===i.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${r.rank} and targets rank ${i.rank}`),p(r.shape.slice(0,r.shape.length-1),i.shape,`predictions's shape should be align with the targets' shape, except the last dimension.`);let a=r.shape[r.shape.length-1];f(n>0&&n<=a,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${a}), but got ${n}`);let o=await r.data(),s=await i.data(),[c,l]=[o.length/a,a],u=T(`bool`,c);for(let e=0;e<c;e++){let t=e*l,r=o.subarray(t,t+l),i=[];for(let e=0;e<r.length;e++)i.push({value:r[e],index:e});i.sort((e,t)=>t.value-e.value),u[e]=0;for(let t=0;t<n;t++)if(i[t].index===s[e]){u[e]=1;break}}return e!==r&&r.dispose(),t!==i&&i.dispose(),Ki(u,i.shape,`bool`)}var ah=ih;function oh(e,t,n,r,i,a=`NHWC`,o){let s=e;e.rank===3&&(s=V(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let c=t;c.rank===3&&(c=V(t,[1,t.shape[0],t.shape[1],t.shape[2]])),f(s.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${s.shape}.`),f(c.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${c.shape}.`),f(n.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${n}.`);let l=a===`NHWC`?s.shape[3]:s.shape[1],u=a===`NHWC`?c.shape[3]:c.shape[1];f(l===n[2],()=>`Error in conv2dDerFilter: depth of input ${l}) must match input depth in filter (${n[2]}.`),f(u===n[3],()=>`Error in conv2dDerFilter: depth of dy (${u}) must match output depth for filter (${n[3]}).`),Dc(`conv2dDerFilter`,i,o);let d={x:s,dy:c},p={strides:r,pad:i,dataFormat:a,dimRoundingMode:o,filterShape:n};return P.runKernel(Qe,d,p)}var sh=I({conv2DBackpropFilter_:oh});function ch(e,t,n){if(n==null||n===`linear`)return e;if(n===`relu`)return B(e,_m(t));throw Error(`Cannot compute gradient for fused activation ${n}.`)}function lh(e,t){let n=t,r=is(e.shape,t.shape);return r.length>0&&(n=ju(n,r)),V(n,e.shape)}function uh(e,t,n,r){if(t===`linear`)return e;if(t===`relu`)return hp(e);if(t===`elu`)return lu(e);if(t===`relu6`)return _p(e);if(t===`prelu`)return Lf(e,n);if(t===`leakyrelu`)return cd(e,r);if(t===`sigmoid`)return Ic(e);throw Error(`Unknown fused activation ${t}.`)}var dh=(e,t)=>!(e>0)||t===`linear`;function fh({x:e,filter:t,strides:n,pad:r,dataFormat:i=`NHWC`,dilations:a=[1,1],dimRoundingMode:o,bias:s,activation:c=`linear`,preluActivationWeights:l,leakyreluAlpha:u}){if(c||=`linear`,dh(P.state.gradientDepth,c)===!1){f(i===`NHWC`,()=>`Error in fused conv2d: got dataFormat of ${i} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let d=yl(e,t,n,r,i,a,o);return s!=null&&(d=Ms(d,s)),uh(d,c,l,u)}let d=F(e,`x`,`conv2d`,`float32`),p=F(t,`filter`,`conv2d`,`float32`),m=d,h=!1;d.rank===3&&(h=!0,m=V(d,[1,d.shape[0],d.shape[1],d.shape[2]])),f(m.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${m.rank}.`),f(p.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${p.rank}.`),Dc(`fused conv2d`,r,o);let g=i===`NHWC`?m.shape[3]:m.shape[1];f(p.shape[2]===g,()=>`Error in conv2d: depth of input (${g}) must match input depth for filter ${p.shape[2]}.`),f(Tc(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`);let _=pc(m.shape,p.shape,n,a,r,o),v;s!=null&&(v=F(s,`bias`,`fused conv2d`),[v]=Ci(v,d),i===`NHWC`?z(_.outShape,v.shape):(f(v.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${v.shape.length}.`),f(v.shape.length===0||v.shape[0]===_.outChannels||v.shape[0]===1,()=>`Error in fused conv2d: bias shape (${v.shape}) is not compatible with the number of output channels (${_.outChannels})`)));let y;if(l!=null){let e=l.shape;if(f(e.length<=1||e.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${e.length}.`),e.length===1)f(e[0]===1||e[0]===_.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${e}) is not compatible with the number of output channels (${_.outChannels}).`);else if(e.length===3)try{z(e,_.outShape)}catch{let t=`Error in fused conv2d: PReLU activation weights (${e}) is not compatible with the output shape of the conv2d (${_.outShape}).`;throw Error(t)}y=F(l,`prelu weights`,`fused conv2d`)}let b=(e,t)=>{f(i===`NHWC`,()=>`Error in gradient of fused conv2D: got dataFormat of ${i} but only NHWC is currently supported.`);let[o,s,l,u]=t,d=ch(e,l,c);f(wc(a),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${a}'`);let p=[Cl(s.shape,d,o,n,r),sh(s,d,o.shape,n,r)];if(u!=null){let e=lh(u,d);p.push(e)}return p},x={x:m,filter:p,bias:v,preluActivationWeights:y},S={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o,activation:c,leakyreluAlpha:u};return s==null?bd((e,t,n)=>{let r=P.runKernel(fr,x,S);return n([t,e,r]),h&&(r=V(r,[r.shape[1],r.shape[2],r.shape[3]])),{value:r,gradFunc:b}})(m,p):bd((e,t,n,r)=>{let i=P.runKernel(fr,x,S);return r([t,e,i,n]),h&&(i=V(i,[i.shape[1],i.shape[2],i.shape[3]])),{value:i,gradFunc:b}})(m,p,v)}var ph=I({fusedConv2d_:fh});function mh(e,t,n,r,i,a=[1,1],o){let s=e;e.rank===3&&(s=V(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let c=t;c.rank===3&&(c=V(t,[1,t.shape[0],t.shape[1],t.shape[2]]));let l={x:s,dy:c},u={strides:r,pad:i,dimRoundingMode:o,dilations:a,filterShape:n};return P.runKernel(ut,l,u)}var hh=I({depthwiseConv2dNativeBackpropFilter_:mh});function gh(e,t,n,r,i,a=[1,1],o){let s=t,c=!1;t.rank===3&&(c=!0,s=V(t,[1,t.shape[0],t.shape[1],t.shape[2]]));let l={dy:s,filter:n},u={strides:r,pad:i,dimRoundingMode:o,dilations:a,inputShape:e},d=P.runKernel(dt,l,u);return c?V(d,[d.shape[1],d.shape[2],d.shape[3]]):d}var _h=I({depthwiseConv2dNativeBackpropInput_:gh});function vh({x:e,filter:t,strides:n,pad:r,dataFormat:i=`NHWC`,dilations:a=[1,1],dimRoundingMode:o,bias:s,activation:c=`linear`,preluActivationWeights:l,leakyreluAlpha:u}){if(dh(P.state.gradientDepth,c)===!1){let d=Gl(e,t,n,r,i,a,o);return s!=null&&(d=Ms(d,s)),uh(d,c,l,u)}let d=F(e,`x`,`depthwiseConv2d`,`float32`),p=F(t,`filter`,`depthwiseConv2d`,`float32`),m=d,h=!1;d.rank===3&&(h=!0,m=V(d,[1,d.shape[0],d.shape[1],d.shape[2]])),f(m.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${m.rank}.`),f(p.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${p.rank}.`),f(m.shape[3]===p.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${m.shape[3]}) must match the inChannels dimension in filter ${p.shape[2]}.`),a??=[1,1],f(Tc(n,a),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),Dc(`fused depthwiseConv2d`,r,o);let g=pc(m.shape,p.shape,n,a,r,o,!0),_;s!=null&&(_=F(s,`bias`,`fused conv2d`),[_]=Ci(_,d),z(g.outShape,_.shape));let v;l!=null&&(v=F(l,`prelu weights`,`fused depthwiseConv2d`));let y=(e,t)=>{f(wc(a),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${a}'`);let[i,s,l,u]=t,d=ch(e,l,c),p=_h(s.shape,d,i,n,r,a,o),m=hh(s,d,i.shape,n,r,a,o);return u==null?[p,m]:[p,m,lh(_,d)]},b={x:m,filter:p,bias:_,preluActivationWeights:v},x={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o,activation:c,leakyreluAlpha:u};return s==null?bd((e,t,n)=>{let r=P.runKernel(pr,b,x);return n([t,e,r]),h&&(r=V(r,[r.shape[1],r.shape[2],r.shape[3]])),{value:r,gradFunc:y}})(m,p):bd((e,t,n,r)=>{let i=P.runKernel(pr,b,x);return r([t,e,i,n]),h&&(i=V(i,[i.shape[1],i.shape[2],i.shape[3]])),{value:i,gradFunc:y}})(m,p,_)}var yh=I({fusedDepthwiseConv2d_:vh});function bh({a:e,b:t,transposeA:n=!1,transposeB:r=!1,bias:i,activation:a=`linear`,preluActivationWeights:o,leakyreluAlpha:s=.2}){if(dh(P.state.gradientDepth,a)===!1){let c=R(e,t,n,r);return i!=null&&(c=Ms(c,i)),uh(c,a,o,s)}let c=F(e,`a`,`fused matMul`),l=F(t,`b`,`fused matMul`);[c,l]=Ci(c,l);let u=n?c.shape[c.rank-2]:c.shape[c.rank-1],d=r?l.shape[l.rank-1]:l.shape[l.rank-2],p=n?c.shape[c.rank-1]:c.shape[c.rank-2],m=r?l.shape[l.rank-2]:l.shape[l.rank-1],h=c.shape.slice(0,-2),_=l.shape.slice(0,-2),v=g(h),y=g(_);f(u===d,()=>`Error in fused matMul: inner shapes (${u}) and (${d}) of Tensors with shapes ${c.shape} and ${l.shape} and transposeA=${n} and transposeB=${r} must match.`);let b=z(c.shape.slice(0,-2),l.shape.slice(0,-2)).concat([p,m]),x=n?V(c,[v,u,p]):V(c,[v,p,u]),S=r?V(l,[y,m,d]):V(l,[y,d,m]),C;i!=null&&(C=F(i,`bias`,`fused matMul`),[C]=Ci(C,c),z(b,C.shape));let w;o!=null&&(w=F(o,`prelu weights`,`fused matMul`));let T=(e,t)=>{let[o,s,c,l]=t,u=ch(V(e,c.shape),c,a),d,f;if(!n&&!r?(d=R(u,s,!1,!0),f=R(o,u,!0,!1)):!n&&r?(d=R(u,s,!1,!1),f=R(u,o,!0,!1)):n&&!r?(d=R(s,u,!1,!0),f=R(o,u,!1,!1)):(d=R(s,u,!0,!0),f=R(u,o,!0,!0)),i!=null){let e=lh(l,u);return[d,f,e]}return[d,f]},E={a:x,b:S,bias:C,preluActivationWeights:w},D={transposeA:n,transposeB:r,activation:a,leakyreluAlpha:s};return i==null?bd((e,t,n)=>{let r=P.runKernel(dr,E,D);return n([e,t,r]),{value:V(r,b),gradFunc:T}})(x,S):bd((e,t,n,r)=>{let i=P.runKernel(dr,E,D);return r([e,t,i,n]),{value:V(i,b),gradFunc:T}})(x,S,C)}var xh=I({fusedMatMul_:bh}),Sh=t({conv2d:()=>ph,depthwiseConv2d:()=>yh,matMul:()=>xh});function Ch(e){return rh(e,.54,.46)}var wh=I({hammingWindow_:Ch});function Th(e){return rh(e,.5,.5)}var Eh=I({hannWindow_:Th});function Dh(e,t,n,r=!1,i=0){let a=0,o=[];for(;a+t<=e.size;)o.push(H(e,a,t)),a+=n;if(r)for(;a<e.size;){let r=a+t-e.size,s=Pc([H(e,a,t-r),sl([r],i)]);o.push(s),a+=n}return o.length===0?Cm([],[0,t]):V(Pc(o),[o.length,t])}var Oh=I({frame_:Dh});function kh(e,t,n,r,i=Eh){return r??=nh(t),lm(B(Oh(e,t,n),i(t)),r)}var Ah=I({stft_:kh});function jh(e,t,n,r,i=`bilinear`,a=0){let o=F(e,`image`,`cropAndResize`),s=F(t,`boxes`,`cropAndResize`,`float32`),c=F(n,`boxInd`,`cropAndResize`,`int32`),l=s.shape[0];f(o.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${o.rank}.`),f(s.rank===2&&s.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${l},4] but had shape ${s.shape}.`),f(c.rank===1&&c.shape[0]===l,()=>`Error in cropAndResize: boxInd must be have size [${l}] but had shape ${s.shape}.`),f(r.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${r.length}.`),f(r[0]>=1&&r[1]>=1,()=>`cropSize must be atleast [1,1], but was ${r}`),f(i===`bilinear`||i===`nearest`,()=>`method must be bilinear or nearest, but was ${i}`);let u={image:o,boxes:s,boxInd:c},d={method:i,extrapolationValue:a,cropSize:r};return P.runKernel(ot,u,d)}var Mh=I({cropAndResize_:jh});function Nh(e){let t=F(e,`image`,`flipLeftRight`,`float32`);f(t.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);let n={image:t};return P.runKernel(Ct,n,{})}var Ph=I({flipLeftRight_:Nh});function Fh(e){let t=F(e,`image`,`grayscaleToRGB`),n=t.rank-1,r=t.shape[n];f(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),f(r===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${r}.`);let i=Array(t.rank);return i.fill(1,0,n),i[n]=3,Wu(t,i)}var Ih=I({grayscaleToRGB_:Fh});function Lh(e,t,n=0,r=.5){let i=F(e,`image`,`rotateWithOffset`,`float32`);f(i.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${i.rank}.`);let a={image:i},o={radians:t,fillValue:n,center:r};return P.runKernel(ur,a,o)}var Rh=I({rotateWithOffset_:Lh});function zh(e,t,n,r,i,a){r??=.5,i??=-1/0,a??=0;let o=e.shape[0];return n=Math.min(n,o),f(0<=r&&r<=1,()=>`iouThreshold must be in [0, 1], but was '${r}'`),f(e.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${e.rank}'`),f(e.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${e.shape[1]}`),f(t.rank===1,()=>`scores must be a 1D tensor`),f(t.shape[0]===o,()=>`scores has incompatible shape with boxes. Expected ${o}, but was ${t.shape[0]}`),f(0<=a&&a<=1,()=>`softNmsSigma must be in [0, 1], but was '${a}'`),{maxOutputSize:n,iouThreshold:r,scoreThreshold:i,softNmsSigma:a}}function Bh(e,t,n,r=.5,i=-1/0){let a=F(e,`boxes`,`nonMaxSuppression`,`float32`),o=F(t,`scores`,`nonMaxSuppression`,`float32`),s=zh(a,o,n,r,i);n=s.maxOutputSize,r=s.iouThreshold,i=s.scoreThreshold;let c={maxOutputSize:n,iouThreshold:r,scoreThreshold:i};return P.runKernel(an,{boxes:a,scores:o},c)}var Vh=I({nonMaxSuppression_:Bh});function Hh(e,t,n){let r=Uh(e,t,n),i=r<0?-(r+1):r;e.splice(i,0,t)}function Uh(e,t,n){return Gh(e,t,n||Wh)}function Wh(e,t){return e>t?1:e<t?-1:0}function Gh(e,t,n){let r=0,i=e.length,a=0,o=!1;for(;r<i;){a=r+(i-r>>>1);let s=n(t,e[a]);s>0?r=a+1:(i=a,o=!s)}return o?r:-r-1}function Kh(e,t,n,r,i){return Yh(e,t,n,r,i,0)}function qh(e,t,n,r,i,a){return Yh(e,t,n,r,i,0,!1,a,!0)}function Jh(e,t,n,r,i,a){return Yh(e,t,n,r,i,a,!0)}function Yh(e,t,n,r,i,a,o=!1,s=!1,c=!1){let l=[];for(let e=0;e<t.length;e++)t[e]>i&&l.push({score:t[e],boxIndex:e,suppressBeginIndex:0});l.sort(Qh);let u=a>0?-.5/a:0,d=[],f=[];for(;d.length<n&&l.length>0;){let t=l.pop(),{score:n,boxIndex:a,suppressBeginIndex:o}=t;if(n<i)break;let s=!1;for(let n=d.length-1;n>=o;--n){let o=Xh(e,a,d[n]);if(o>=r){s=!0;break}if(t.score*=Zh(r,u,o),t.score<=i)break}t.suppressBeginIndex=d.length,s||(t.score===n?(d.push(a),f.push(t.score)):t.score>i&&Hh(l,t,Qh))}let p=d.length,m=n-p;s&&m>0&&(d.push(...Array(m).fill(0)),f.push(...Array(m).fill(0)));let h={selectedIndices:d};return o&&(h.selectedScores=f),c&&(h.validOutputs=p),h}function Xh(e,t,n){let r=e.subarray(t*4,t*4+4),i=e.subarray(n*4,n*4+4),a=Math.min(r[0],r[2]),o=Math.min(r[1],r[3]),s=Math.max(r[0],r[2]),c=Math.max(r[1],r[3]),l=Math.min(i[0],i[2]),u=Math.min(i[1],i[3]),d=Math.max(i[0],i[2]),f=Math.max(i[1],i[3]),p=(s-a)*(c-o),m=(d-l)*(f-u);if(p<=0||m<=0)return 0;let h=Math.max(a,l),g=Math.max(o,u),_=Math.min(s,d),v=Math.min(c,f),y=Math.max(_-h,0)*Math.max(v-g,0);return y/(p+m-y)}function Zh(e,t,n){let r=Math.exp(t*n*n);return n<=e?r:0}function Qh(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}async function $h(e,t,n,r=.5,i=-1/0){let a=F(e,`boxes`,`nonMaxSuppressionAsync`),o=F(t,`scores`,`nonMaxSuppressionAsync`),s=zh(a,o,n,r,i);n=s.maxOutputSize,r=s.iouThreshold,i=s.scoreThreshold;let c=await Promise.all([a.data(),o.data()]),l=c[0],u=c[1],{selectedIndices:d}=Kh(l,u,n,r,i);return a!==e&&a.dispose(),o!==t&&o.dispose(),Sm(d,`int32`)}var eg=$h;function tg(e,t,n,r=.5,i=-1/0,a=0){let o=F(e,`boxes`,`nonMaxSuppression`),s=F(t,`scores`,`nonMaxSuppression`),c=zh(o,s,n,r,i,a);n=c.maxOutputSize,r=c.iouThreshold,i=c.scoreThreshold,a=c.softNmsSigma;let l={boxes:o,scores:s},u={maxOutputSize:n,iouThreshold:r,scoreThreshold:i,softNmsSigma:a},d=P.runKernel(sn,l,u);return{selectedIndices:d[0],selectedScores:d[1]}}var ng=I({nonMaxSuppressionWithScore_:tg});async function rg(e,t,n,r=.5,i=-1/0,a=0){let o=F(e,`boxes`,`nonMaxSuppressionAsync`),s=F(t,`scores`,`nonMaxSuppressionAsync`),c=zh(o,s,n,r,i,a);n=c.maxOutputSize,r=c.iouThreshold,i=c.scoreThreshold,a=c.softNmsSigma;let l=await Promise.all([o.data(),s.data()]),u=l[0],d=l[1],{selectedIndices:f,selectedScores:p}=Jh(u,d,n,r,i,a);return o!==e&&o.dispose(),s!==t&&s.dispose(),{selectedIndices:Sm(f,`int32`),selectedScores:Sm(p)}}var ig=rg;function ag(e,t,n,r=.5,i=-1/0,a=!1){let o=F(e,`boxes`,`nonMaxSuppression`),s=F(t,`scores`,`nonMaxSuppression`),c=zh(o,s,n,r,i,null),l=c.maxOutputSize,u=c.iouThreshold,d=c.scoreThreshold,f={boxes:o,scores:s},p={maxOutputSize:l,iouThreshold:u,scoreThreshold:d,padToMaxOutputSize:a},m=P.runKernel(on,f,p);return{selectedIndices:m[0],validOutputs:m[1]}}var og=I({nonMaxSuppressionPadded_:ag});async function sg(e,t,n,r=.5,i=-1/0,a=!1){let o=F(e,`boxes`,`nonMaxSuppressionAsync`),s=F(t,`scores`,`nonMaxSuppressionAsync`),c=zh(o,s,n,r,i,null),l=c.maxOutputSize,u=c.iouThreshold,d=c.scoreThreshold,[f,p]=await Promise.all([o.data(),s.data()]),{selectedIndices:m,validOutputs:h}=qh(f,p,l,u,d,a);return o!==e&&o.dispose(),s!==t&&s.dispose(),{selectedIndices:Sm(m,`int32`),validOutputs:U(h,`int32`)}}var cg=sg;function lg(e,t,n=!1,r=!1){let i=F(e,`images`,`resizeBilinear`);f(i.rank===3||i.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${i.rank}.`),f(t.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),f(r===!1||n===!1,()=>`Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.`);let a=i,o=!1;i.rank===3&&(o=!0,a=V(i,[1,i.shape[0],i.shape[1],i.shape[2]]));let[]=t,s={images:a},c={alignCorners:n,halfPixelCenters:r,size:t},l=P.runKernel(Cn,s,c);return o?V(l,[l.shape[1],l.shape[2],l.shape[3]]):l}var ug=I({resizeBilinear_:lg});function dg(e,t,n=!1,r=!1){let i=F(e,`images`,`resizeNearestNeighbor`);f(i.rank===3||i.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${i.rank}.`),f(t.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),f(i.dtype===`float32`||i.dtype===`int32`,()=>"`images` must have `int32` or `float32` as dtype"),f(r===!1||n===!1,()=>`Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.`);let a=i,o=!1;i.rank===3&&(o=!0,a=V(i,[1,i.shape[0],i.shape[1],i.shape[2]]));let[]=t,s={images:a},c={alignCorners:n,halfPixelCenters:r,size:t},l=P.runKernel(xn,s,c);return o?V(l,[l.shape[1],l.shape[2],l.shape[3]]):l}var fg=I({resizeNearestNeighbor_:dg});function pg(e,t=`binary`,n=!1,r=.5){let i=F(e,`image`,`threshold`),a=i.shape[0]*i.shape[1],o=B(Sm([r]),255),s,c,l,u;if(f(i.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${i.rank}.`),f(i.shape[2]===3||i.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${i.shape[2]}.`),f(i.dtype===`int32`||i.dtype===`float32`,()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${i.dtype}.`),f(t===`otsu`||t===`binary`,()=>`Method must be binary or otsu, but was ${t}`),i.shape[2]===3){[s,c,l]=sm(i,[1,1,1],-1);let e=B(s,.2989),t=B(c,.587),n=B(l,.114);u=Ms(Ms(e,t),n)}else u=e;return t===`otsu`&&(o=mg(el(io(kp(u),`int32`),Ki([]),256),a)),io(B(n?fd(u,o):Qu(u,o),255),`int32`)}function mg(e,t){let n=Sm([-1]),r=Sm([0]),i=Sm([0]),a,o,s,c,l,u;for(let d=0;d<e.size-1;d++){a=H(e,0,d+1),o=H(e,d+1),l=Is(ju(a),t),u=Is(ju(o),t),s=Is(ju(B(a,dp(0,a.size))),ju(a));let f=sl(o.shape,a.size),p=Ms(dp(0,o.size),f);c=Is(ju(B(o,p)),ju(o));let m=W(s,c),h=W(s,c);i=B(B(B(l,u),m),h);let g=Qu(i,r);r=$l(g,i,r),n=$l(g,Sm([d]),n)}return n}var hg=I({threshold_:pg});function gg(e,t,n=`nearest`,r=`constant`,i=0,a){let o=F(e,`image`,`transform`,`float32`),s=F(t,`transforms`,`transform`,`float32`);f(o.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${o.rank}.`),f(s.rank===2&&(s.shape[0]===o.shape[0]||s.shape[0]===1)&&s.shape[1]===8,()=>`Error in transform: Input transform should be batch x 8 or 1 x 8`),f(a==null||a.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${a}.`);let c={image:o,transforms:s},l={interpolation:n,fillMode:r,fillValue:i,outputShape:a};return P.runKernel(nr,c,l)}var _g=I({transform_:gg});function vg(e,t,n){f(t%1==0,()=>`bandPart(): numLower must be an integer, got ${t}.`),f(n%1==0,()=>`bandPart(): numUpper must be an integer, got ${n}.`);let r=F(e,`a`,`bandPart`);f(r.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${r.rank}.`);let i=r.shape,[a,o]=r.shape.slice(-2);if(!(t<=a))throw Error(`bandPart(): numLower (${t}) must not be greater than the number of rows (${a}).`);if(!(n<=o))throw Error(`bandPart(): numUpper (${n}) must not be greater than the number of columns (${o}).`);t<0&&(t=a),n<0&&(n=o);let s=W(V(dp(0,a,1,`int32`),[-1,1]),dp(0,o,1,`int32`)),c=jd(fd(s,U(+t,`int32`)),ed(s,U(-n,`int32`))),l=Qd([a,o],r.dtype);return V(hm(Im(V(r,[-1,a,o])).map(e=>$l(c,e,l))),i)}var yg=I({bandPart_:vg});function bg(e){let t;if(Array.isArray(e)){t=!1,f(e!=null&&e.length>0,()=>`Gram-Schmidt process: input must not be null, undefined, or empty`);let n=e[0].shape[0];for(let t=1;t<e.length;++t)f(e[t].shape[0]===n,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${e[t].shape[0]} vs. ${n})`)}else t=!0,e=sm(e,e.shape[0],0).map(e=>pm(e,[0]));f(e.length<=e[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${e.length}) exceeds number of dimensions (${e[0].shape[0]}).`);let n=[],r=e;for(let t=0;t<e.length;++t)n.push(P.tidy(()=>{let e=r[t];if(t>0)for(let r=0;r<t;++r){let t=B(ju(B(n[r],e)),n[r]);e=W(e,t)}return Is(e,Pu(e,`euclidean`))}));return t?hm(n,0):n}var xg=I({gramSchmidt_:bg});function Sg(e,t=!1){if(f(e.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${e.rank}`),e.rank===2)return Cg(e,t);{let n=Im(V(e,[e.shape.slice(0,e.shape.length-2).reduce((e,t)=>e*t),e.shape[e.shape.length-2],e.shape[e.shape.length-1]]),0),r=[],i=[];return n.forEach(e=>{let[n,a]=Cg(e,t);r.push(n),i.push(a)}),[V(hm(r,0),e.shape),V(hm(i,0),e.shape)]}}function Cg(e,t=!1){return P.tidy(()=>{f(e.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${e.shape.length}D Tensor.`);let n=e.shape[0],r=e.shape[1],i=Ku(n),a=oo(e),o=Cm([[1]],[1,1]),s=oo(o),c=n>=r?r:n;for(let e=0;e<c;++e){let t=a,c=s,l=i;[s,a,i]=P.tidy(()=>{let t=H(a,[e,e],[n-e,1]),c=Pu(t),l=H(a,[e,e],[1,1]),u=$l(Qu(l,0),Cm([[-1]]),Cm([[1]])),d=W(l,B(u,c)),f=Is(t,d);s=f.shape[0]===1?oo(o):Pc([o,H(f,[1,0],[f.shape[0]-1,f.shape[1]])],0);let p=Qo(Is(R(u,d),c)),m=H(a,[e,0],[n-e,r]),h=B(p,s),g=ns(s);if(e===0)a=W(m,R(h,R(g,m)));else{let t=W(m,R(h,R(g,m)));a=Pc([H(a,[0,0],[e,r]),t],0)}let _=ns(h),v=H(i,[0,e],[n,i.shape[1]-e]);if(e===0)i=W(v,R(R(v,s),_));else{let t=W(v,R(R(v,s),_));i=Pc([H(i,[0,0],[n,e]),t],1)}return[s,a,i]}),Ho([t,c,l])}return!t&&n>r&&(i=H(i,[0,0],[n,r]),a=H(a,[0,0],[r,r])),[i,a]})}var wg=I({qr_:Sg}),Tg;(function(e){e[e.NONE=0]=`NONE`,e[e.MEAN=1]=`MEAN`,e[e.SUM=2]=`SUM`,e[e.SUM_BY_NONZERO_WEIGHTS=3]=`SUM_BY_NONZERO_WEIGHTS`})(Tg||={});function Eg(e,t,n=Tg.SUM_BY_NONZERO_WEIGHTS){let r=F(e,`losses`,`computeWeightedLoss`),i=null;t!=null&&(i=F(t,`weights`,`computeWeightedLoss`));let a=i==null?r:B(r,i);if(n===Tg.NONE)return a;if(n===Tg.SUM)return ju(a);if(n===Tg.MEAN){if(i==null)return Zd(a);{let e=r.size/i.size,t=Is(ju(a),ju(i));return e>1?Is(t,U(e)):t}}if(n===Tg.SUM_BY_NONZERO_WEIGHTS){if(i==null)return Is(ju(a),U(r.size));{let e=io(ju(hf(B(i,$d(r.shape)),U(0))),`float32`);return Is(ju(a),e)}}throw Error(`Unknown reduction: ${n}`)}var Dg=I({computeWeightedLoss_:Eg});function Og(e,t,n,r=Tg.SUM_BY_NONZERO_WEIGHTS){let i=F(e,`labels`,`absoluteDifference`),a=F(t,`predictions`,`absoluteDifference`),o=null;return n!=null&&(o=F(n,`weights`,`absoluteDifference`)),p(i.shape,a.shape,`Error in absoluteDifference: `),Dg(zs(W(i,a)),o,r)}var kg=I({absoluteDifference_:Og});function Ag(e,t,n,r,i=Tg.SUM_BY_NONZERO_WEIGHTS){let a=F(e,`labels`,`cosineDistance`),o=F(t,`predictions`,`cosineDistance`),s=null;return r!=null&&(s=F(r,`weights`,`cosineDistance`)),p(a.shape,o.shape,`Error in cosineDistance: `),Dg(W(U(1),ju(B(a,o),n,!0)),s,i)}var jg=I({cosineDistance_:Ag});function Mg(e,t,n,r=Tg.SUM_BY_NONZERO_WEIGHTS){let i=F(e,`labels`,`hingeLoss`),a=F(t,`predictions`,`hingeLoss`),o=null;n!=null&&(o=F(n,`weights`,`hingeLoss`)),p(i.shape,a.shape,`Error in hingeLoss: `);let s=U(1);return i=W(B(U(2),i),s),Dg(hp(W(s,B(i,a))),o,r)}var Ng=I({hingeLoss_:Mg});function Pg(e,t,n,r=1,i=Tg.SUM_BY_NONZERO_WEIGHTS){let a=F(e,`labels`,`huberLoss`),o=F(t,`predictions`,`huberLoss`),s=null;n!=null&&(s=F(n,`weights`,`huberLoss`)),p(a.shape,o.shape,`Error in huberLoss: `);let c=U(r),l=zs(W(o,a)),u=nf(l,c),d=W(l,u);return Dg(Ms(B(U(.5),ku(u)),B(c,d)),s,i)}var Fg=I({huberLoss_:Pg});function Ig(e,t,n,r=1e-7,i=Tg.SUM_BY_NONZERO_WEIGHTS){let a=F(e,`labels`,`logLoss`),o=F(t,`predictions`,`logLoss`),s=null;n!=null&&(s=F(n,`weights`,`logLoss`)),p(a.shape,o.shape,`Error in logLoss: `);let c=U(1),l=U(r);return Dg(W(Qo(B(a,_d(Ms(o,l)))),B(W(c,a),_d(Ms(W(c,o),l)))),s,i)}var Lg=I({logLoss_:Ig});function Rg(e,t,n,r=Tg.SUM_BY_NONZERO_WEIGHTS){let i=F(e,`labels`,`meanSquaredError`),a=F(t,`predictions`,`meanSquaredError`),o=null;return n!=null&&(o=F(n,`weights`,`meanSquaredError`)),p(i.shape,a.shape,`Error in meanSquaredError: `),Dg(dm(i,a),o,r)}var zg=I({meanSquaredError_:Rg});function Bg(e,t){let n=F(e,`labels`,`sigmoidCrossEntropyWithLogits`),r=F(t,`logits`,`sigmoidCrossEntropyWithLogits`);p(n.shape,r.shape,`Error in sigmoidCrossEntropyWithLogits: `);let i=hp(r),a=B(r,n),o=yd(Ru(Qo(zs(r))));return Ms(W(i,a),o)}function Vg(e,t,n,r=0,i=Tg.SUM_BY_NONZERO_WEIGHTS){let a=F(e,`multiClassLabels`,`sigmoidCrossEntropy`),o=F(t,`logits`,`sigmoidCrossEntropy`),s=null;if(n!=null&&(s=F(n,`weights`,`sigmoidCrossEntropy`)),p(a.shape,o.shape,`Error in sigmoidCrossEntropy: `),r>0){let e=U(r),t=U(1),n=U(.5);a=Ms(B(a,W(t,e)),B(n,e))}return Dg(Bg(a,o),s,i)}var Hg=I({sigmoidCrossEntropy_:Vg});function Ug(e,t,n=-1){if(n===-1&&(n=t.rank-1),n!==t.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${t.rank} and dim was ${n}`);return bd((e,t,r)=>{let i=kd(t,[n],!0),a=W(io(t,`float32`),i);return r([e,a]),{value:ju(Qo(B(a,e)),[n]),gradFunc:(e,t)=>{let[r,i]=t,a=hu(e.shape,[n]);return[B(V(e,a),W(io(r,`float32`),Ru(i))),B(V(e,a),W(Ru(i),io(r,`float32`)))]}}})(e,t)}function Wg(e,t,n,r=0,i=Tg.SUM_BY_NONZERO_WEIGHTS){let a=F(e,`onehotLabels`,`softmaxCrossEntropy`),o=F(t,`logits`,`softmaxCrossEntropy`),s=null;if(n!=null&&(s=F(n,`weights`,`softmaxCrossEntropy`)),p(a.shape,o.shape,`Error in softmaxCrossEntropy: `),r>0){let e=U(r),t=U(1),n=U(a.shape[1]);a=Ms(B(a,W(t,e)),Is(e,n))}return Dg(Ug(a,o),s,i)}var Gg=I({softmaxCrossEntropy_:Wg});function Kg(e,t,n,r){let i=F(e,`indices`,`sparseFillEmptyRows`,`int32`),a=F(t,`values`,`sparseFillEmptyRows`),o=F(n,`denseShape`,`sparseFillEmptyRows`,`int32`),s=F(r,`defaultValue`,`sparseFillEmptyRows`,a.dtype);if(i.rank!==2)throw Error(`Indices should be Tensor2D but received shape
        ${i.shape}`);if(a.rank!==1)throw Error(`Values should be Tensor1D but received shape ${a.shape}`);if(o.rank!==1)throw Error(`Dense shape should be Tensor1D but received shape ${o.shape}`);if(s.rank!==0)throw Error(`Default value should be a scalar but received shape ${s.shape}`);let c={indices:i,values:a,denseShape:o,defaultValue:s},l=P.runKernel(Hn,c);return{outputIndices:l[0],outputValues:l[1],emptyRowIndicator:l[2],reverseIndexMap:l[3]}}var qg=I({sparseFillEmptyRows_:Kg});function Jg(e,t,n){let r=F(e,`inputIndices`,`sparseReshape`,`int32`),i=F(t,`inputShape`,`sparseReshape`,`int32`),a=F(n,`newShape`,`sparseReshape`,`int32`);if(r.rank!==2)throw Error(`Input indices should be Tensor2D but received shape
        ${r.shape}`);if(i.rank!==1)throw Error(`Input shape should be Tensor1D but received shape ${i.shape}`);if(a.rank!==1)throw Error(`New shape should be Tensor1D but received shape ${a.shape}`);let o={inputIndices:r,inputShape:i,newShape:a},s=P.runKernel(Un,o);return{outputIndices:s[0],outputShape:s[1]}}var Yg=I({sparseReshape_:Jg});function Xg(e,t,n){let r=F(e,`data`,`sparseSegmentMean`),i=F(t,`indices`,`sparseSegmentMean`,`int32`),a=F(n,`segmentIds`,`sparseSegmentMean`,`int32`);if(r.rank<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.rank!==1)throw Error(`Indices should be Tensor1D but received shape
          ${i.shape}`);if(a.rank!==1)throw Error(`Segment ids should be Tensor1D but received shape
          ${a.shape}`);let o={data:r,indices:i,segmentIds:a};return P.runKernel(Wn,o)}var Zg=I({sparseSegmentMean_:Xg});function Qg(e,t,n){let r=F(e,`data`,`sparseSegmentSum`),i=F(t,`indices`,`sparseSegmentSum`,`int32`),a=F(n,`segmentIds`,`sparseSegmentSum`,`int32`);if(r.rank<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.rank!==1)throw Error(`Indices should be Tensor1D but received shape
         ${i.shape}`);if(a.rank!==1)throw Error(`Segment ids should be Tensor1D but received shape
         ${a.shape}`);let o={data:r,indices:i,segmentIds:a};return P.runKernel(Gn,o)}var $g=I({sparseSegmentSum_:Qg});function e_(e,t,n,r,i,a,o,s){let c=F(e,`data`,`stringNGrams`,`string`);if(c.dtype!==`string`)throw Error(`Data must be of datatype string`);if(c.shape.length!==1)throw Error(`Data must be a vector, saw: ${c.shape}`);let l=F(t,`dataSplits`,`stringNGrams`);if(l.dtype!==`int32`)throw Error(`Data splits must be of datatype int32`);let u={separator:n,nGramWidths:r,leftPad:i,rightPad:a,padWidth:o,preserveShortSequences:s},d={data:c,dataSplits:l},f=P.runKernel(Xn,d,u);return{nGrams:f[0],nGramsSplits:f[1]}}var t_=I({stringNGrams_:e_});function n_(e,t,n=!0){let r=F(e,`input`,`stringSplit`,`string`),i=F(t,`delimiter`,`stringSplit`,`string`);if(r.rank!==1)throw Error(`Input should be Tensor1D but received shape ${r.shape}`);if(i.rank!==0)throw Error(`Delimiter should be a scalar but received shape ${i.shape}`);let a={skipEmpty:n},o={input:r,delimiter:i},s=P.runKernel(Zn,o,a);return{indices:s[0],values:s[1],shape:s[2]}}var r_=I({stringSplit_:n_});function i_(e,t){let n=F(e,`input`,`stringToHashBucketFast`,`string`),r={numBuckets:t};if(t<=0)throw Error(`Number of buckets must be at least 1`);let i={input:n};return P.runKernel(Qn,i,r)}var a_=I({stringToHashBucketFast_:i_}),o_={fft:tm,ifft:rm,rfft:lm,irfft:am},s_={hammingWindow:wh,hannWindow:Eh,frame:Oh,stft:Ah},c_={flipLeftRight:Ph,grayscaleToRGB:Ih,resizeNearestNeighbor:fg,resizeBilinear:ug,rotateWithOffset:Rh,cropAndResize:Mh,nonMaxSuppression:Vh,nonMaxSuppressionAsync:eg,nonMaxSuppressionWithScore:ng,nonMaxSuppressionWithScoreAsync:ig,nonMaxSuppressionPadded:og,nonMaxSuppressionPaddedAsync:cg,threshold:hg,transform:_g},l_={bandPart:yg,gramSchmidt:xg,qr:wg},u_={absoluteDifference:kg,computeWeightedLoss:Dg,cosineDistance:jg,hingeLoss:Ng,huberLoss:Fg,logLoss:Lg,meanSquaredError:zg,sigmoidCrossEntropy:Hg,softmaxCrossEntropy:Gg},d_={sparseFillEmptyRows:qg,sparseReshape:Yg,sparseSegmentMean:Zg,sparseSegmentSum:$g},f_={stringNGrams:t_,stringSplit:r_,stringToHashBucketFast:a_},p_=typeof requestAnimationFrame<`u`?requestAnimationFrame:typeof setImmediate<`u`?setImmediate:e=>e();function m_(){return new Promise(e=>p_(()=>e()))}function h_(e,t){let n=e[0].length;e.forEach((e,t)=>{f(e.length===n,()=>`Error in concat${n}D: rank of tensors[${t}] must be the same as the rank of the rest (${n})`)}),f(t>=0&&t<n,()=>`Error in concat${n}D: axis must be between 0 and ${n-1}.`);let r=e[0];e.forEach((e,i)=>{for(let a=0;a<n;a++)f(a===t||e[a]===r[a],()=>`Error in concat${n}D: Shape of tensors[${i}] (${e}) does not match the shape of the rest (${r}) along the non-concatenated axis ${i}.`)})}function g_(e,t){let n=e[0].slice();for(let r=1;r<e.length;r++)n[t]+=e[r][t];return n}var __;(function(e){e[e.FIRST_DIM_SIZE=0]=`FIRST_DIM_SIZE`,e[e.VALUE_ROWIDS=1]=`VALUE_ROWIDS`,e[e.ROW_LENGTHS=2]=`ROW_LENGTHS`,e[e.ROW_SPLITS=3]=`ROW_SPLITS`,e[e.ROW_LIMITS=4]=`ROW_LIMITS`,e[e.ROW_STARTS=5]=`ROW_STARTS`})(__||={});function v_(e,t,n){let r=[];if(n==null&&t==null)return r;if(t==null)for(;r.length<e+n.length;)r.push(-1);else r=t.slice();if(n==null)return r;if(e+n.length!==r.length)throw Error(`rt input.shape and shape=${t} are incompatible: rt input.rank = ${e+n.length}, but shape.rank = ${r.length}`);for(let i=1;i<n.length;++i){let a=n[i],o=r[r.length-n.length+i],s=r[o];if(a>=0){if(s>=0){if(s!==a)throw Error(`rt input.shape and shape=${t} are incompatible: rt input.shape[${i+e}] = ${a} but shape[${i+e}] = ${s}`)}else r[o]=a}}return r}function y_(e){let t={FIRST_DIM_SIZE:__.FIRST_DIM_SIZE,VALUE_ROWIDS:__.VALUE_ROWIDS,ROW_LENGTHS:__.ROW_LENGTHS,ROW_SPLITS:__.ROW_SPLITS,ROW_LIMITS:__.ROW_LIMITS,ROW_STARTS:__.ROW_STARTS},n=[];for(let r of e)if(r in t)n.push(t[r]);else break;return n}function b_(e){return e.length===0?0:e[0]===__.FIRST_DIM_SIZE?e.length-1:e.length}function x_(e,t){if(e==null||t==null)return;let n=e.length,r=t.length;if(n>=r)throw Error(`defaultValue.shape=${e} and ragged tensor flatValues.shape=${t}, are incompatible: defaultValue.rank = ${n} must be less than ragged tensor input flatValues.rank = ${r})`);for(let i=0;i<Math.min(n,r-1);++i){let n=e[i],r=t[i+1];if(n>=0&&r>=0&&n!==1&&n!==r)throw Error(`defaultValue.shape=${e}, and ragged tensor input flatValues.shape=${t} are incompatible: defaultValue.shape[${i-e.length}] = ${n} but ragged tensor input.flatValues.shape[${i-e.length}] = ${r}`)}}function S_(e){return e<=30?e:se(e,Math.floor(Math.sqrt(e)))}function C_(e,t,n){return[n*(typeof e==`number`?e:e[0]),t*(typeof e==`number`?e:e[1])]}function w_(e,t,n,r=!0){let i=[];if(r)i=i.concat(t.slice(0)),i.push(e[0]/n),i=i.concat(e.slice(1));else{i=i.concat(e[0]);let n=t.length;for(let r=0;r<n;++r)i=i.concat([e[r+1]/t[r],t[r]]);i=i.concat(e.slice(n+1))}return i}function T_(e,t,n=!0){let r=[];if(n){r.push(t);for(let n=t+1;n<e;++n)n<=2*t?(r.push(n),r.push(n-(t+1))):r.push(n)}else{let n=[],i=[];for(let r=1;r<e;++r)r>=t*2+1||r%2==1?i.push(r):n.push(r);r.push(...n),r.push(0),r.push(...i)}return r}function E_(e,t,n,r=!0){let i=[];r?i.push(e[0]/n):i.push(e[0]*n);for(let n=1;n<e.length;++n)n<=t.length?r?i.push(t[n-1]*e[n]):i.push(e[n]/t[n-1]):i.push(e[n]);return i}function D_(e,t){let n=[0];for(let r=0;r<t;++r)n.push(e[r][0]);return n}function O_(e,t,n){let r=e.slice(0,1);for(let i=0;i<n;++i)r.push(e[i+1]-t[i][0]-t[i][1]);return r}var k_=1.7580993408473768,A_=1.0507009873554805,j_=.3275911,M_=.254829592,N_=-.284496736,P_=1.421413741,F_=-1.453152027,I_=1.061405429;function L_(e,t){if(e.length!==t.length)throw Error(`Cannot merge real and imag arrays of different lengths. real:${e.length}, imag: ${t.length}.`);let n=new Float32Array(e.length*2);for(let r=0;r<n.length;r+=2)n[r]=e[r/2],n[r+1]=t[r/2];return n}function R_(e){let t=new Float32Array(e.length/2),n=new Float32Array(e.length/2);for(let r=0;r<e.length;r+=2)t[r/2]=e[r],n[r/2]=e[r+1];return{real:t,imag:n}}function z_(e){let t=Math.ceil(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let t=0;t<e.length;t+=4)n[Math.floor(t/4)]=e[t],r[Math.floor(t/4)]=e[t+1];return{real:n,imag:r}}function B_(e){let t=Math.floor(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let t=2;t<e.length;t+=4)n[Math.floor(t/4)]=e[t],r[Math.floor(t/4)]=e[t+1];return{real:n,imag:r}}function V_(e,t){return{real:e[t*2],imag:e[t*2+1]}}function H_(e,t,n,r){e[r*2]=t,e[r*2+1]=n}function U_(e,t){let n=new Float32Array(e/2),r=new Float32Array(e/2);for(let i=0;i<Math.ceil(e/2);i++){let a=(t?2:-2)*Math.PI*(i/e);n[i]=Math.cos(a),r[i]=Math.sin(a)}return{real:n,imag:r}}function W_(e,t,n){let r=(n?2:-2)*Math.PI*(e/t);return{real:Math.cos(r),imag:Math.sin(r)}}var G_=`->`,K_=/->/g,q_=`,`,J_=`...`;function Y_(e,t){e=e.replace(/\s/g,``);let n=(e.length-e.replace(K_,``).length)/2;if(n<1)throw Error(`Equations without an arrow are not supported.`);if(n>1)throw Error(`Equation must contain exactly one arrow ("${G_}").`);let[r,i]=e.split(G_);f(r.indexOf(J_)===-1,()=>`The ellipsis notation ("${J_}") is not supported yet.`);let a=r.split(q_),o=a.length;if(t!==o)throw Error(`Expected ${o} input tensors, received ${t}`);if(o>2)throw Error(`Support for more than 2 input tensors is not implemented yet.`);let s=[];for(let e=0;e<i.length;++e){let t=i[e];if(!a.some(e=>e.indexOf(t)!==-1))throw Error(`Output subscripts contain the label ${t} not present in the input subscripts.`);s.indexOf(t)===-1&&s.push(t)}for(let e=0;e<r.length;++e){let t=r[e];s.indexOf(t)===-1&&t!==q_&&s.push(t)}let c=Array(a.length);for(let e=0;e<o;++e){if(new Set(a[e].split(``)).size!==a[e].length)throw Error(`Found duplicate axes in input component ${a[e]}. Support for duplicate axes in input is not implemented yet.`);c[e]=[];for(let t=0;t<a[e].length;++t)c[e].push(s.indexOf(a[e][t]))}let l=s.length,u=i.length,d=[];for(let e=u;e<l;++e)d.push(e);return{allDims:s,summedDims:d,idDims:c}}function X_(e,t){let n=Array(e);n.fill(-1);for(let e=0;e<t.length;++e)n[t[e]]=e;let r=[];for(let t=0;t<e;++t)n[t]===-1&&r.push(t);return n=n.filter(e=>e!==-1),{permutationIndices:n,expandDims:r}}function Z_(e,t,n){let r=Array(e);for(let e=0;e<n.length;++e){let i=n[e].shape;for(let n=0;n<t[e].length;++n)r[t[e][n]]===void 0?r[t[e][n]]=i[n]:f(r[t[e][n]]===i[n],()=>`Expected dimension ${r[t[e][n]]} at axis ${n} of input shaped ${JSON.stringify(i)}, but got dimension ${i[n]}`)}}function Q_(e,t){let n=e,r=[],i=0;e.length===0&&n.push(-1),i=e.length+1;for(let e=0;e<i;++e)r.push([]);let a=[];for(let e=0;e<n.length;++e){let i=n[e],o=ev(t,i);for(let t of o)a.indexOf(t)===-1&&(r[e].push(t),a.push(t))}return{path:n,steps:r}}function $_(e){return e.every((e,t)=>e===t)}function ev(e,t){let n=[];for(let r=0;r<e.length;++r)(e[r].length===0||e[r].indexOf(t)!==-1||t===-1)&&n.push(r);return n}function tv(e,t,n=0){let r=[];if(typeof t==`number`)f(e.shape[n]%t===0,()=>`Number of splits must evenly divide the axis.`),r=Array(t).fill(e.shape[n]/t);else{f(t.reduce((e,t)=>(t===-1&&(e+=1),e),0)<=1,()=>`There should be only one negative value in split array.`);let i=t.indexOf(-1);if(i!==-1){let r=t.reduce((e,t)=>t>0?e+t:e);t[i]=e.shape[n]-r}f(e.shape[n]===t.reduce((e,t)=>e+t),()=>`The sum of sizes must match the size of the axis dimension.`),r=t}return r}function nv(e){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${e}`}function rv(e,t){return`indices(${e}, 0) is invalid: ${t} < 0`}function iv(e,t,n){return`indices(${e}, 0) is invalid: ${t} >= ${n}`}function av(e,t){return`only one output dimension may be -1, not both ${e} and ${t}`}function ov(e,t){return`size ${e} must be non-negative, not ${t}`}function sv(){return`reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero`}function cv(e,t){return`Input to reshape is a SparseTensor with ${g(e)}
  dense values, but the requested shape requires a multiple of ${g(t)}. inputShape=${e} outputShape= ${t}`}function lv(e,t){return`Input to reshape is a tensor with ${g(e)} dense values, but the requested shape has ${g(t)}. inputShape=${e} outputShape=${t}`}function uv(){return`segment ids must be >= 0`}function dv(){return`segment ids are not increasing`}function fv(e,t){return`Segment id ${e} out of range [0, ${t}), possibly because segmentIds input is not sorted.`}function pv(e,t,n){return`Bad: indices[${e}] == ${t} out of range [0, ${n})`}var mv=t({collectGatherOpShapeInfo:()=>_v,computeOutShape:()=>gv,segOpComputeOptimalWindowSize:()=>hv});function hv(e,t){let n=!1,r;for(e<=30?(r=e,n=!0):r=se(e,Math.floor(Math.sqrt(e)));!n;)r>t||r===e?n=!0:r=se(e,r+1);return r}function gv(e,t,n){let r=[],i=e.length;for(let a=0;a<i;a++)a===t?r.push(n):r.push(e[a]);return r}function _v(e,t,n,r){let i=t.shape.length,a=e.shape.length;if(r!==0&&(r<-i||r>i))throw Error(`Expect batchDims in the range of [-${i}, ${i}], but got ${r}`);if(r<0&&(r+=i),r>a)throw Error(`batchDims (${r}) must be less than rank(x) (
    ${a}).`);if(n<r)throw Error(`batchDims (${r}) must be less than or equal to axis (${n}).`);for(let n=0;n<r;++n)if(e.shape[n]!==t.shape[n])throw Error(`x.shape[${n}]: ${e.shape[n]} should be equal to indices.shape[${n}]: ${t.shape[n]}.`);let o=e.shape[n],s=[],c=1,l=1,u=1;for(let t=0;t<r;++t)s.push(e.shape[t]),c*=e.shape[t];for(let t=r;t<n;t++)s.push(e.shape[t]),l*=e.shape[t];for(let e=r;e<i;e++)s.push(t.shape[e]);for(let t=n+1;t<a;t++)s.push(e.shape[t]),u*=e.shape[t];return{batchSize:c,sliceSize:u,outerSize:l,dimSize:o,outputShape:s}}var vv=t({ERF_A1:()=>M_,ERF_A2:()=>N_,ERF_A3:()=>P_,ERF_A4:()=>F_,ERF_A5:()=>I_,ERF_P:()=>j_,PARALLELIZE_THRESHOLD:()=>30,RowPartitionType:()=>__,SELU_SCALE:()=>A_,SELU_SCALEALPHA:()=>k_,applyActivation:()=>uh,assertAndGetBroadcastShape:()=>z,assertAxesAreInnerMostDims:()=>gu,assertParamsConsistent:()=>h_,assignToTypedArray:()=>H_,axesAreInnerMostDims:()=>fu,calculateShapes:()=>ls,checkEinsumDimSizes:()=>Z_,checkPadOnDimRoundingMode:()=>Dc,combineLocations:()=>pu,combineRaggedTensorToTensorShapes:()=>v_,complexWithEvenIndex:()=>z_,complexWithOddIndex:()=>B_,computeConv2DInfo:()=>pc,computeConv3DInfo:()=>mc,computeDefaultPad:()=>_c,computeDilation2DInfo:()=>uc,computeOptimalWindowSize:()=>S_,computeOutAndReduceShapes:()=>mu,computeOutShape:()=>g_,computePool2DInfo:()=>dc,computePool3DInfo:()=>fc,convertConv2DDataFormat:()=>Ec,decodeEinsumEquation:()=>Y_,eitherStridesOrDilationsAreOne:()=>Tc,expandShapeToKeepDim:()=>hu,exponent:()=>W_,exponents:()=>U_,fromStringArrayToUint8:()=>bv,fromUint8ToStringArray:()=>yv,getAxesPermutation:()=>_u,getBroadcastDims:()=>rs,getComplexWithIndex:()=>V_,getEinsumComputePath:()=>Q_,getEinsumPermutation:()=>X_,getFusedBiasGradient:()=>lh,getFusedDyActivation:()=>ch,getImageCenter:()=>C_,getInnerMostAxes:()=>yu,getPermuted:()=>T_,getRaggedRank:()=>b_,getReductionAxes:()=>is,getReshaped:()=>w_,getReshapedPermuted:()=>E_,getRowPartitionTypesHelper:()=>y_,getSliceBeginCoords:()=>D_,getSliceSize:()=>O_,getSparseFillEmptyRowsIndicesDenseShapeMismatch:()=>nv,getSparseFillEmptyRowsNegativeIndexErrorMessage:()=>rv,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:()=>iv,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:()=>sv,getSparseReshapeInputOutputMismatchErrorMessage:()=>lv,getSparseReshapeInputOutputMultipleErrorMessage:()=>cv,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:()=>av,getSparseReshapeNegativeOutputDimErrorMessage:()=>ov,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:()=>pv,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:()=>uv,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:()=>dv,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:()=>fv,getUndoAxesPermutation:()=>vu,isIdentityPermutation:()=>$_,log:()=>hr,mergeRealAndImagArrays:()=>L_,prepareAndValidate:()=>os,prepareSplitSize:()=>tv,segment_util:()=>mv,shouldFuse:()=>dh,slice_util:()=>us,splitRealAndImagArrays:()=>R_,tupleValuesAreOne:()=>wc,upcastType:()=>xi,validateDefaultValueShape:()=>x_,validateInput:()=>cs,validateUpdateShape:()=>ss,warn:()=>mr});function yv(e){try{return e.map(e=>Kr(e))}catch(e){throw Error(`Failed to decode encoded string bytes into utf-8, error: ${e}`)}}function bv(e){return e.map(e=>Gr(e))}function G(e,t){Array.isArray(e)||(e=[e]),e.forEach(e=>{e!=null&&f(e.dtype!==`complex64`,()=>`${t} does not support complex64 tensors in the CPU backend.`)})}var xv=zm,Sv=class e extends o{constructor(){super(),this.blockSize=48,this.firstUse=!0,this.data=new a(this,Bo())}nextDataId(){return e.nextDataId++}write(e,t,n){this.firstUse&&(this.firstUse=!1,N().get(`IS_NODE`)&&mr(`
============================
Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details. 
============================`));let r={id:this.nextDataId()};return this.data.set(r,{values:e,dtype:n,refCount:1}),r}makeTensorInfo(e,t,n){let r;if(t===`string`&&n!=null&&n.length>0&&ne(n[0])){let i=n.map(e=>Gr(e));r=this.write(i,e,t)}else r=this.write(n,e,t);return{dataId:r,shape:e,dtype:t}}refCount(e){return this.data.has(e)?this.data.get(e).refCount:0}incRef(e){let t=this.data.get(e);t.refCount++}decRef(e){if(this.data.has(e)){let t=this.data.get(e);t.refCount--}}move(e,t,n,r,i){this.data.set(e,{values:t,dtype:r,refCount:i})}numDataIds(){return this.data.numDataIds()}async read(e){return this.readSync(e)}readSync(e){let{dtype:t,complexTensorInfos:n}=this.data.get(e);return t===`complex64`?L_(this.readSync(n.real.dataId),this.readSync(n.imag.dataId)):this.data.get(e).values}bufferSync(e){let t=this.readSync(e.dataId);if(e.dtype===`string`)try{let n=t.map(e=>Kr(e));return L(e.shape,e.dtype,n)}catch{throw Error(`Failed to decode encoded string bytes into utf-8`)}return L(e.shape,e.dtype,t)}makeOutput(e,t,n){return Bo().makeTensorFromTensorInfo(this.makeTensorInfo(t,n,e),this)}disposeData(e,t=!1){if(this.data.has(e)){if(this.data.get(e).refCount--,!t&&this.data.get(e).refCount>0)return!1;let{complexTensorInfos:n}=this.data.get(e);n!=null&&(this.disposeData(n.real.dataId,!0),this.disposeData(n.imag.dataId,!0)),this.data.delete(e)}return!0}disposeIntermediateTensorInfo(e){this.disposeData(e.dataId)}async time(e){let t=Wr();return e(),{kernelMs:Wr()-t}}memory(){return{unreliable:!0,reasons:[`The reported memory is an upper bound. Due to automatic garbage collection, the true allocated memory may be less.`]}}where(e){G([e],`where`);let t=this.readSync(e.dataId);return xv(e.shape,t)}dispose(){}floatPrecision(){return 32}epsilon(){return super.epsilon()}};Sv.nextDataId=0;function Cv(e){let t=new Float32Array(e.length);for(let n=0;n<e.length;++n)t[n]=Math.abs(e[n]);return t}var wv={kernelName:`Abs`,backendName:`cpu`,kernelFunc:e=>{let{x:t}=e.inputs,n=e.backend;G(t,`abs`);let r=new Float32Array(g(t.shape)),i=n.data.get(t.dataId).values;return r=Cv(i),n.makeOutput(r,t.shape,t.dtype)}};function Tv(e){return(t,n,r,i,a)=>{let o=z(t,n),s=o.length,c=j(o),l=T(a,g(o)),u=t.length,d=n.length,f=j(t),p=j(n),m=rs(t,o),h=rs(n,o);if(m.length+h.length===0)for(let t=0;t<l.length;++t)l[t]=e(r[t%r.length],i[t%i.length]);else for(let t=0;t<l.length;++t){let n=me(t,s,c),a=n.slice(-u);m.forEach(e=>a[e]=0);let o=pe(a,u,f),g=n.slice(-d);h.forEach(e=>g[e]=0);let _=pe(g,d,p);l[t]=e(r[o],i[_])}return[l,o]}}function Ev(e){let{inputs:t,backend:n}=e,{real:r,imag:i}=t,a=n.data.get(r.dataId).values,o=n.data.get(i.dataId).values,s=n.makeTensorInfo(r.shape,`complex64`),c=n.data.get(s.dataId);return c.complexTensorInfos={real:n.makeTensorInfo(r.shape,`float32`,a),imag:n.makeTensorInfo(i.shape,`float32`,o)},s}var Dv={kernelName:Je,backendName:`cpu`,kernelFunc:Ev};function Ov(e,t,n=`float32`){if(n===`complex64`)return Ev({inputs:{real:Ov(e,t,`float32`),imag:Ov(e,t,`float32`)},backend:e});let r=M(g(t),n);return e.makeTensorInfo(t,n,r)}function kv(e){let{inputs:t,backend:n}=e,{x:r}=t;return n.incRef(r.dataId),{dataId:r.dataId,shape:r.shape,dtype:r.dtype}}var Av={kernelName:jt,backendName:`cpu`,kernelFunc:kv};function jv(e){let{inputs:t,backend:n}=e,{input:r}=t,i=n.data.get(r.dataId).complexTensorInfos.real,a=n.data.get(i.dataId).values;return n.makeTensorInfo(i.shape,i.dtype,a)}var Mv={kernelName:_n,backendName:`cpu`,kernelFunc:jv};function Nv(e,t,n,r){if(r===`int32`)return[t,`int32`,Int32Array.from(e)];if(r===`bool`){let r=Ur([0],n),[i,a]=Tv((e,t)=>e===t?0:1)(t,[],e,r,`bool`);return[a,`bool`,i]}throw Error(`Error in Cast: failed to cast ${n} to ${r}`)}function Pv(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dtype:a}=r;if(a===`complex64`){if(i.dtype===`complex64`)return kv({inputs:{x:i},backend:n});let e=Ov(n,i.shape,i.dtype),t=Pv({inputs:{x:i},backend:n,attrs:{dtype:`float32`}}),r=Ev({inputs:{real:t,imag:e},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),r}if(i.dtype===`complex64`){let e=jv({inputs:{input:i},backend:n}),t=Pv({inputs:{x:e},backend:n,attrs:{dtype:a}});return n.disposeIntermediateTensorInfo(e),t}if(!ee(i.dtype,a)){let e=kv({inputs:{x:i},backend:n});return{dataId:e.dataId,shape:e.shape,dtype:a}}let o=n.data.get(i.dataId).values,[s,c,l]=Nv(o,i.shape,i.dtype,a);return n.makeTensorInfo(s,c,l)}var Fv={kernelName:Ge,backendName:`cpu`,kernelFunc:Pv};function Iv(e,t,n,r){return n==null?({inputs:n,backend:i})=>{let{a,b:o}=n,s=i;G([a,o],e);let c=s.data.get(a.dataId).values,l=s.data.get(o.dataId).values,u=a.dtype===`string`?yv(c):c,d=a.dtype===`string`?yv(l):l,f=r||a.dtype,[p,m]=t(a.shape,o.shape,u,d,f);return s.makeTensorInfo(m,f,p)}:({inputs:e,backend:i})=>{let{a,b:o}=e,s=i;if(a.dtype===`complex64`||o.dtype===`complex64`){let e=Pv({inputs:{x:a},backend:s,attrs:{dtype:`complex64`}}),t=s.data.get(e.dataId),r=t.complexTensorInfos.real,i=t.complexTensorInfos.imag,c=s.data.get(r.dataId).values,l=s.data.get(i.dataId).values,u=Pv({inputs:{x:o},backend:s,attrs:{dtype:`complex64`}}),d=s.data.get(u.dataId),f=d.complexTensorInfos.real,p=d.complexTensorInfos.imag,m=s.data.get(f.dataId).values,h=s.data.get(p.dataId).values,[g,_,v]=n(a.shape,o.shape,c,l,m,h),y=s.makeTensorInfo(v,`float32`,g),b=s.makeTensorInfo(v,`float32`,_),x=Ev({inputs:{real:y,imag:b},backend:s});return s.disposeIntermediateTensorInfo(e),s.disposeIntermediateTensorInfo(u),s.disposeIntermediateTensorInfo(y),s.disposeIntermediateTensorInfo(b),x}{let e=s.data.get(a.dataId).values,n=s.data.get(o.dataId).values,i=r||a.dtype,[c,l]=t(a.shape,o.shape,e,n,i);return s.makeTensorInfo(l,i,c)}}}function Lv(e){return(t,n,r,i,a,o)=>{let s=z(t,n),c=g(s),l=s.length,u=j(s),d=T(`float32`,c),f=T(`float32`,c),p=rs(t,s),m=rs(n,s),h=L_(r,i),_=L_(a,o),v=t.length,y=j(t),b=n.length,x=j(n);if(p.length+m.length===0)for(let t=0;t<d.length;t++){let n=t%h.length,r=t%_.length,i=e(h[n*2],h[n*2+1],_[r*2],_[r*2+1]);d[t]=i.real,f[t]=i.imag}else for(let t=0;t<d.length;t++){let n=me(t,l,u),r=n.slice(-v);p.forEach(e=>r[e]=0);let i=pe(r,v,y),a=n.slice(-b);m.forEach(e=>a[e]=0);let o=pe(a,b,x),s=e(h[i*2],h[i*2+1],_[o*2],_[o*2+1]);d[t]=s.real,f[t]=s.imag}return[d,f,s]}}var Rv=Tv(((e,t)=>e+t)),zv=Iv(`Add`,Rv,Lv(((e,t,n,r)=>({real:e+n,imag:t+r})))),Bv={kernelName:`Add`,backendName:`cpu`,kernelFunc:zv};function Vv(e,t,n,r,i){let a=g(r),o=M(i,n);for(let n=0;n<e.length;n++){let r=e[n];if(r<0)throw Error(`Input x must be non-negative!`);r>=i||(a>0?o[r]+=t[n]:o[r]+=1)}return o}function Hv(e,t,n,r=!1){let i=e.shape[0],a=e.shape[1],o=L([i,n],t.dtype);for(let s=0;s<i;s++)for(let i=0;i<a;i++){let a=e.get(s,i);if(a<0)throw Error(`Input x must be non-negative!`);a>=n||(r?o.set(1,s,a):t.size>0?o.set(o.get(s,a)+t.get(s,i),s,a):o.set(o.get(s,a)+1,s,a))}return o}function Uv(e){return(t,n,r)=>{let i=T(n,t.length);for(let n=0;n<t.length;++n)i[n]=e(t[n],r);return i}}function K(e,t,n){return({inputs:r,attrs:i,backend:a})=>{let{x:o}=r;if(G(o,e),o.dtype===`string`||n===`string`)throw Error(`unaryKernelFunc does not support string input/output`);let s=a,c=s.data.get(o.dataId).values,l=g(o.shape),u=n||o.dtype,d=E(u,l);for(let e=0;e<l;++e)d[e]=t(c[e],i);return s.makeTensorInfo(o.shape,u,d)}}function Wv(e,t,n){return({inputs:r,attrs:i,backend:a})=>{let{x:o}=r;if(G(o,e),o.dtype===`string`||n===`string`)throw Error(`unaryKernelFunc does not support string input/output`);let s=a,c=s.data.get(o.dataId).values,l=n||o.dtype,u=t(c,l,i);return s.makeTensorInfo(o.shape,l,u)}}var Gv=Uv(e=>Math.ceil(e)),Kv={kernelName:Ke,backendName:`cpu`,kernelFunc:Wv(Ke,Gv)};function qv(e,t,n,r){let i=E(n,g(t));if(r&&n!==`string`){let t=0;e.forEach(e=>{let n=g(e.shape);i.set(e.vals,t),t+=n})}else{let r=0;e.forEach(e=>{let a=n===`string`?yv(e.vals):e.vals,o=0;for(let n=0;n<e.shape[0];++n){let s=n*t[1]+r;for(let t=0;t<e.shape[1];++t)i[s+t]=a[o++]}r+=e.shape[1]})}return i}var Jv=Tv((e,t)=>+(e===t)),Yv=Iv(yt,Jv,null,`bool`),Xv={kernelName:yt,backendName:`cpu`,kernelFunc:Yv},Zv=Uv(e=>Math.exp(e)),Qv=Wv(`Exp`,Zv,`float32`),$v={kernelName:`Exp`,backendName:`cpu`,kernelFunc:Qv},ey=Uv(e=>Math.expm1(e)),ty={kernelName:xt,backendName:`cpu`,kernelFunc:Wv(xt,ey)},ny=Uv(e=>Math.floor(e)),ry={kernelName:wt,backendName:`cpu`,kernelFunc:Wv(wt,ny)};function iy(e,t,n,r,i,a,o,s,c){let l=L([r,a],n);for(let n=0;n<r;n++){let r=[],u=0;for(let t=0;t<i;t++){let a=e[n*i+t];u+=a*o[t],r.push(a)}if(u<0||u>=c/a)throw Error(`Invalid indices: ${r} does not index into ${s}`);for(let e=0;e<a;e++)l.values[n*a+e]=t.get(...t.indexToLoc(u*a+e))}return l}function ay(e,t,n){let r=L(n,e.dtype);for(let n=0;n<r.size;++n){let i=r.indexToLoc(n).slice(),a=i[0],o=i[2],s=t.locToIndex([a,o]);i[2]=t.values[s];let c=e.locToIndex(i);0<=c&&c<e.values.length&&(r.values[n]=e.values[c])}return r}var oy=Tv((e,t)=>+(e>t)),sy={kernelName:kt,backendName:`cpu`,kernelFunc:Iv(kt,oy,null,`bool`)},cy=Tv((e,t)=>+(e>=t)),ly={kernelName:At,backendName:`cpu`,kernelFunc:Iv(At,cy,null,`bool`)},uy=Tv((e,t)=>+(e<t)),dy={kernelName:Rt,backendName:`cpu`,kernelFunc:Iv(Rt,uy,null,`bool`)},fy=Tv((e,t)=>+(e<=t)),py={kernelName:zt,backendName:`cpu`,kernelFunc:Iv(zt,fy,null,`bool`)};function my(e,t,n){let r=(t-e)/(n-1),i=M(n,`float32`);i[0]=e;for(let e=1;e<i.length;e++)i[e]=i[e-1]+r;return i}var hy=Uv(e=>Math.log(e)),gy={kernelName:`Log`,backendName:`cpu`,kernelFunc:Wv(`Log`,hy)};function _y(e,t,n,r){let i=T(r,g(n));for(let n=0;n<i.length;++n){let r=n*t,a=e[r];for(let n=0;n<t;++n){let t=e[r+n];(Number.isNaN(t)||t>a)&&(a=t)}i[n]=a}return i}var vy=Tv(((e,t)=>Math.max(e,t))),yy={kernelName:Kt,backendName:`cpu`,kernelFunc:Iv(Kt,vy)},by=Tv(((e,t)=>Math.min(e,t))),xy={kernelName:$t,backendName:`cpu`,kernelFunc:Iv($t,by)},Sy=Tv(((e,t)=>e*t)),Cy=Iv(nn,Sy,Lv(((e,t,n,r)=>({real:e*n-t*r,imag:e*r+t*n})))),wy={kernelName:nn,backendName:`cpu`,kernelFunc:Cy};function Ty(e,t,n){return Sy([],t,Vr(-1,n),e,n)}function Ey(e){let{inputs:t,backend:n}=e,{x:r}=t;G(r,`neg`);let i=n.data.get(r.dataId).values,[a,o]=Ty(i,r.shape,r.dtype);return n.makeTensorInfo(o,r.dtype,a)}var Dy={kernelName:`Neg`,backendName:`cpu`,kernelFunc:Ey},Oy=Tv(((e,t)=>e===t?0:1)),ky={kernelName:rn,backendName:`cpu`,kernelFunc:Iv(rn,Oy,null,`bool`)};function Ay(e,t,n,r,i){let a=t.length,o=g(t),s=j(t),c=j(i),l=T(n,g(i));for(let t=0;t<o;++t){let n=me(t,a,s),i=Array(n.length);for(let e=0;e<i.length;e++)i[e]=n[r[e]];let o=pe(i,a,c);l[o]=e[t]}return l}function jy(e){let{inputs:t,attrs:n,backend:r}=e,{x:i}=t,{perm:a}=n;G(i,`transpose`);let o=i.shape.length,s=Array(o);for(let e=0;e<s.length;e++)s[e]=i.shape[a[e]];let c=r.data.get(i.dataId).values,l=Ay(c,i.shape,i.dtype,a,s);return{dataId:r.write(l,s,i.dtype),shape:s,dtype:i.dtype}}var My={kernelName:rr,backendName:`cpu`,kernelFunc:jy};function Ny(e,t,n,r){let[i,a]=mu(e,r),o=xi(t,`int32`),s=M(g(i),o),c=g(a);for(let e=0;e<s.length;++e){let t=e*c,r=1;for(let e=0;e<c;++e)r*=n[t+e];s[e]=r}return{outVals:s,outShape:i,outDtype:o}}function Py(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;G(i,`prod`);let s=i.shape.length,c=C(a,i.shape),l=_u(c,s),u=c,d=i,f=[];l!=null&&(d=jy({inputs:{x:i},backend:n,attrs:{perm:l}}),f.push(d),u=yu(u.length,s));let p=n.data.get(d.dataId).values,{outVals:m,outShape:h,outDtype:g}=Ny(d.shape,d.dtype,p,u),_=h;return o&&(_=hu(h,c)),f.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(_,g,m)}var Fy={kernelName:pn,backendName:`cpu`,kernelFunc:Py};function Iy(e,t,n){e.forEach((e,r)=>{if(e<0||e>=n){let i=me(r,t.length,j(t)).join(`,`);throw Error(`indices[${i}] = ${e} is not in [0, ${n})`)}})}function Ly(e,t){for(let n=0;n<e.length;++n){let r=e[n],i=n===e.length-1?t:e[n+1].length;if(r.length===0)throw Error(`Ragged splits may not be empty`);if(r[0]<0)throw Error(`Ragged splits must be non-negative`);if(r[r.length-1]>i)throw Error(`Ragged splits must not point past values`);for(let e=1;e<r.length;++e)if(r[e-1]>r[e])throw Error(`Ragged splits must be sorted in ascending order`)}}function Ry(e,t,n,r){let i=[],a=0,o=t.length-1+n.length,s=Array(o).fill(null).map(()=>[0]);Ly(n,r);let c=1;for(let e=0;e<t.length-1;++e){c*=t[e];let n=t[e+1];for(let t=1;t<c+1;++t)s[e].push(t*n)}for(let r=0;r<e.length;++r){let o=e[r],c=e[r]+1;for(let e=0;e<n.length;++e){let r=n[e],i=e+t.length-1;if(i>=0){let e=s[i],t=e[e.length-1]-r[o];for(let e=o;e<c;++e)s[i].push(r[e+1]+t)}o=r[o],c=r[c]}c!==o&&(i.push([o,c]),a+=c-o)}return{outSplits:s,valueSlices:i,numValues:a}}function zy(e){let t=[];for(let n=0;n<e.length;++n){let r=e[n].length,i=E(`int32`,r);t.push(i),e[n].forEach((e,t)=>i[t]=e)}return t}function By(e,t){let n=e.slice(0,t);for(;n.length<t;)n.push(1);for(let r=t;r<e.length;r++)n[t-1]*=e[r];return n}function Vy(e,t,n,r,i,a){let o=By(t,2)[1],s=By(a,2)[1],c=0;for(let t of n)for(let n=t[0];n<t[1];++n){for(let t=0;t<r;++t)i[c*s+t]=e[n*o+t];++c}}function Hy(e,t,n,r,i){let a=t.slice();a[0]=i;let o=E(n,g(a)),s=e.length;return Vy(e,t,r,s===0?0:s/t[0],o,a),[o,a]}function Uy(e,t,n,r,i,a,o,s){if(e.length===0)throw Error(`paramsNestedSplits must be non empty`);if(t[0].length===0)throw Error(`Split tensors must not be scalars`);if(Iy(a,o,t[0][0]-1),r.length===0)throw Error(`params.rank must be nonzero`);let c=r[0],{outSplits:l,valueSlices:u,numValues:d}=Ry(a,o,e,c),f=zy(l),p=Hy(n,r,i,u,d);return[f,p[0],p[1]]}var Wy=__,Gy=class e{constructor(e,t,n,r,i,a,o,s,c,l){this.shape=e,this.shapeShape=t,this.values=n,this.valuesShape=r,this.valuesDType=i,this.defaultValue=a,this.defaultValueShape=o,this.rowPartitionValues=s,this.rowPartitionValuesShapes=c,this.rowPartitionTypes=y_(l),this.raggedRank=b_(this.rowPartitionTypes)}getRowPartitionTypeByDimension(e){return this.rowPartitionTypes[0]===Wy.FIRST_DIM_SIZE?this.rowPartitionTypes[e+1]:this.rowPartitionTypes[e]}getRowPartitionTensor(e){return this.rowPartitionTypes[0]===Wy.FIRST_DIM_SIZE?this.rowPartitionValues[e+1]:this.rowPartitionValues[e]}getMaxWidth(t){let n=this.getRowPartitionTensor(t-1);switch(this.getRowPartitionTypeByDimension(t-1)){case Wy.VALUE_ROWIDS:return e.getMaxWidthValueRowID(n);case Wy.ROW_SPLITS:return e.getMaxWidthRowSplit(n);default:throw Error(`Cannot handle partition type ${Wy[this.getRowPartitionTypeByDimension(t-1)]}`)}}static getMaxWidthRowSplit(e){let t=e.length;if(t===0||t===1)return 0;let n=0;for(let r=0;r<t-1;++r){let t=e[r+1]-e[r];t>n&&(n=t)}return n}static getMaxWidthValueRowID(e){let t=e.length;if(t===0)return 0;let n=0,r=e[0],i=0;for(let a=1;a<t;++a){let t=e[a];t!==r&&(r=t,i=Math.max(a-n,i),n=a)}return Math.max(t-n,i)}tensorShapeFromTensor(e,t,n=!0){if(t.length===0){if(e[0]===-1)return[];throw Error(`The only valid scalar shape tensor is the fully unknown shape specified as -1.`)}return qy(e,n)}calculateOutputSize(e){let t=this.valuesShape,n=this.defaultValueShape;x_(n,t);let r=this.tensorShapeFromTensor(this.shape,this.shapeShape),i=v_(this.raggedRank,r,t);i[0]<0&&(i[0]=e);for(let e=1;e<=this.raggedRank;++e)i[e]<0&&(i[e]=this.getMaxWidth(e));return i}calculateFirstParentOutputIndex(e,t,n){let r=Math.min(e,n),i=[],a=0;for(let e=0;e<r;++e,a+=t)i.push(a);for(let t=r;t<e;++t)i.push(-1);return f(i.length===e,()=>`Final length of result must be equal to firstDimension.`),i}calculateOutputIndexRowSplit(e,t,n,r){let i=e.length,a=[];for(let o=0;o<i-1;++o){let i=e[o+1]-e[o],s=Math.min(r,i),c=t[o];c===-1&&(s=0);for(let e=0;e<s;++e)a.push(c),c+=n;for(let e=0;e<i-s;++e)a.push(-1)}if(i>0&&a.length!==e[i-1])throw Error(`Invalid row split size.`);return a}calculateOutputIndexValueRowID(e,t,n,r){let i=e.length,a=[];if(i===0)return[];let o=0,s=e[0];if(s>=t.length)throw Error(`Got currentValueRowId=${s}, which is not less than ${t.length}`);let c=t[s];a.push(c);for(let l=1;l<i;++l){let i=e[l];if(i===s)c>=0&&(++o,o<r?c+=n:c=-1);else{if(o=0,s=i,i>=t.length)throw Error(`Got nextValueRowId=${i} which is not less than ${t.length}`);c=t[i]}a.push(c)}if(a.length!==e.length)throw Error(`Invalid row ids.`);return a}calculateOutputIndex(e,t,n,r){let i=this.getRowPartitionTensor(e),a=this.getRowPartitionTypeByDimension(e);switch(a){case Wy.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(i,t,n,r);case Wy.ROW_SPLITS:if(i.length-1>t.length)throw Error(`Row partition size is greater than output size: ${i.length-1} > ${t.length}`);return this.calculateOutputIndexRowSplit(i,t,n,r);default:throw Error(`Unsupported partition type: ${Wy[a]}`)}}getFirstDimensionSize(){let e=this.rowPartitionValues[0];if(this.rowPartitionTypes.length===0)throw Error(`No row_partition_types given.`);let t=this.rowPartitionTypes[0];switch(t){case Wy.FIRST_DIM_SIZE:return e[0];case Wy.VALUE_ROWIDS:throw Error(`Cannot handle VALUE_ROWIDS in first dimension.`);case Wy.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw Error(`Cannot handle type ${Wy[t]}`)}}compute(){if(this.rowPartitionValues[0].length<=0)throw Error(`Invalid first partition input. Tensor requires at least one element.`);let e=this.getFirstDimensionSize(),t=this.calculateOutputSize(e),n=Array(this.raggedRank+1);n[n.length-1]=1;for(let e=n.length-2;e>=0;--e)n[e]=n[e+1]*t[e+1];let r=qy(t,!1),i=E(this.valuesDType,g(r));if(n[0]*t[0]>0){let a=this.calculateFirstParentOutputIndex(e,n[0],t[0]);for(let e=1;e<=this.raggedRank;++e)a=this.calculateOutputIndex(e-1,a,n[e],t[e]);this.setOutput(this.raggedRank,a,i,r)}return[r,i]}setOutput(e,t,n,r){if(n.length===0)return;let i=this.values,a=n,o=r.slice();o=o.slice(e+1);let s=g(o),c=t.length,l=this.defaultValue;if(l.length!==s&&l.length!==1){let e=this.defaultValueShape;Vo(()=>{l=il(V(l,e),o).dataSync()})}let u=0,d=0,f=0;for(let e=0;e<=c;++e){let r=e<c?t[e]:-1;if(r===f)++f;else{if(d<f){let e=i.subarray(u*s);Ky(a.subarray(d*s),e,(f-d)*s)}if(e>=c){let e=n.length;r=Math.floor(e/s)}if(r>f){if(this.defaultValue.length===1)a.subarray(f*s,r*s).fill(this.defaultValue[0]),f=r;else for(;r>f;)Ky(a.slice(f*s),l,s),++f}r<0?(u=e+1,d=f):(u=e,d=f,f=d+1)}}}};function Ky(e,t,n){for(let r=0;r<n;r++)e[r]=t[r]}function qy(e,t){let n=[];for(let r of e){if(r<0){if(!t)throw Error(`Dimension ${r} must be >= 0`);if(r<-1)throw Error(`Dimension ${r} must be >= -1`);r=-1}n.push(r)}return n}function Jy(e,t,n,r,i,a,o,s,c,l){return new Gy(e,t,n,r,i,a,o,s,c,l).compute()}function Yy(e,t,n,r){if(e===t||e<t&&n<0||t<e&&n>1)return M(0,r);let i=M(Math.abs(Math.ceil((t-e)/n)),r);t<e&&n===1&&(n=-1),i[0]=e;for(let e=1;e<i.length;e++)i[e]=i[e-1]+n;return i}var Xy=Uv(e=>1/Math.sqrt(e)),Zy={kernelName:On,backendName:`cpu`,kernelFunc:Wv(On,Xy)};function Qy(e,t,n,r,i,a,o,s,c,l){let u=[r/i,i],d=e.values,f=t.values;if(r===0)return L(n,t.dtype);let p=L(u,t.dtype);typeof c==`string`||typeof c==`number`?p.values.fill(c):typeof c==`boolean`&&p.values.fill(+c);for(let e=0;e<a;e++){let a=[],c=0;for(let t=0;t<o;t++){let n=d[e*o+t];a.push(n),c+=n*s[t]}if(c<0||c>=r/i)throw Error(`Invalid indices: ${a} does not index into ${n}`);for(let n=0;n<i;n++)l?p.values[c*i+n]+=f[e*i+n]:p.values[c*i+n]=t.rank===0?f[0]:f[e*i+n]}return p}var $y=Uv(e=>1/(1+Math.exp(-e))),eb=K(In,e=>1/(1+Math.exp(-e))),tb={kernelName:In,backendName:`cpu`,kernelFunc:eb};function nb(e,t,n,r,i){let a=Ts(r,t,n),o=g(n),s=j(r);if(a){let n=Es(t,s);return i===`string`?e.slice(n,n+o):e.subarray(n,n+o)}let c=L(r,i,i===`string`?yv(e):e),l=L(n,i);for(let e=0;e<l.size;++e){let n=l.indexToLoc(e),r=n.map((e,n)=>e+t[n]);l.set(c.get(...r),...n)}return i===`string`?bv(l.values):l.values}function rb(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{begin:a,size:o}=r;G(i,`slice`);let[s,c]=Ds(i,a,o);ps(i,s,c);let l=n.data.get(i.dataId).values,u=nb(l,s,c,i.shape,i.dtype);return n.makeTensorInfo(c,i.dtype,u)}var ib={kernelName:Nn,backendName:`cpu`,kernelFunc:rb};function ab(e,t,n,r,i,a,o){let s=t[0],c=a[0],l=Array(c),u=Array(s),d=t[1];if(c===0){if(s!==0)throw Error(nv(s));let e=E(n,0),t=E(i,0);return[e,[0,d],t,l,u]}let f=!0,p=0,m=Array(c).fill(0);for(let t=0;t<s;++t){let n=e[t*d];if(n<0)throw Error(rv(t,n));if(n>=c)throw Error(iv(t,n,c));++m[n],f&&=n>=p,p=n}let h=!0;for(let e=0;e<c;++e){let t=m[e]===0;l[e]=t,h&&=!t,m[e]=Math.max(m[e],1),e>0&&(m[e]+=m[e-1])}if(h&&f){let t=e,n=r;for(let e=0;e<s;++e)u[e]=e;return[t,[s,d],n,l,u]}{let t=m[c-1],a=E(n,t*d),f=E(i,t),p=Array(c).fill(0);for(let t=0;t<s;++t){let n=e[t*d],i=p[n],o=(n===0?0:m[n-1])+i;p[n]++;for(let n=0;n<d;++n)a[o*d+n]=e[t*d+n];f[o]=r[t],u[t]=o}for(let e=0;e<c;++e)if(p[e]===0){let t=e===0?0:m[e-1];a[t*d+0]=e;for(let e=1;e<d;++e)a[t*d+e]=0;f[t]=o}return[a,[t,d],f,l,u]}}function ob(e,t,n,r,i){let a=g(r),o=t[0],s=i.length,c=[],l=1,u=-1;for(let e=0;e<s;++e){let t=i[e];if(t===-1){if(u!==-1)throw Error(av(u,e));u=e,c.push(1)}else{if(t<0)throw Error(ov(e,t));l*=t,c.push(t)}}if(u!==-1){if(l<=0)throw Error(sv());let e=Math.trunc(a/l);if(l*e!==a)throw Error(cv(r,c));c[u]=e}if(g(c)!==a)throw Error(lv(r,c));let d=r.length,f=[];if(d>0){f[d-1]=1;for(let e=d-2;e>=0;--e)f[e]=f[e+1]*r[e+1]}let p=[];if(s>0){p[s-1]=1;for(let e=s-2;e>=0;--e)p[e]=p[e+1]*c[e+1]}let m=E(n,o*s);for(let t=0;t<o;++t){let n=0;for(let r=0;r<d;++r)n+=e[t*d+r]*f[r];for(let e=0;e<s;++e)m[t*s+e]=Math.trunc(n/p[e]),n%=p[e]}return[m,[o,s],c]}function sb(e,t,n,r,i,a=!1,o=0){let s=r.length,c=[t[0],e.length/t[0]],l=c[1],u=s>0?i[s-1]+1:0;if(u<0)throw Error(uv());let d=t.slice();d[0]=u;let f=E(n,d.reduce((e,t)=>e*t,1));if(s===0)return u>0&&f.fill(o),[f,d];if(u<=0)throw Error(uv());let p=0,m=1,h=0,g=i[p];for(;;){let t=0;if(m<s){if(t=i[m],g===t){++m;continue}if(g>=t)throw Error(dv())}if(g<0||g>=u)throw Error(fv(g,u));g>h&&f.fill(o,h*l,g*l);for(let t=p;t<m;++t){let n=r[t];if(n<0||n>=c[0])throw Error(pv(t,r[t],c[0]));for(let t=0;t<l;t++)f[g*l+t]+=e[n*l+t]}if(a)for(let e=0;e<l;e++)f[g*l+e]/=m-p;if(p=m,++m,h=g+1,g=t,m>s)break}return h<u&&f.fill(o,h*l,u*l),[f,d]}var cb=Uv(e=>Math.sqrt(e)),lb={kernelName:Rn,backendName:`cpu`,kernelFunc:K(Rn,e=>Math.sqrt(e))},ub=Tv(((e,t)=>{let n=e-t;return n*n})),db={kernelName:qn,backendName:`cpu`,kernelFunc:Iv(qn,ub)};function fb(e,t,n,r){let i=L(e,t.dtype);for(let e=0;e<i.size;e++){let a=i.indexToLoc(e),o=Array(a.length);for(let e=0;e<o.length;e++)o[e]=a[e]*n[e]+r[e];i.set(t.get(...o),...a)}return i}var pb=class{constructor(e,t,n,r,i,a){this.separator=Gr(e),this.nGramWidths=t,this.leftPad=Gr(n),this.rightPad=Gr(r),this.padWidth=i,this.preserveShort=a}getPadWidth(e){return Math.min(this.padWidth<0?e-1:this.padWidth,e-1)}getNumNGrams(e,t){let n=this.getPadWidth(t);return Math.max(0,e+2*n-t+1)}createNGrams(e,t,n,r,i,a){for(let o=0;o<i;++o){let s=this.getPadWidth(a),c=Math.max(0,s-o),l=Math.max(0,s-(i-(o+1))),u=a-(c+l),d=t+(c>0?0:o-s),f=0;f+=c*this.leftPad.length;for(let t=0;t<u;++t)f+=e[d+t].length;f+=l*this.rightPad.length;let p=c+l+u-1;f+=p*this.separator.length,n[r+o]=new Uint8Array(f);let m=n[r+o],h=0,g=e=>e.forEach(e=>m[h++]=e);for(let e=0;e<c;++e)g(this.leftPad),g(this.separator);for(let t=0;t<u-1;++t)g(e[d+t]),g(this.separator);if(u>0){g(e[d+u-1]);for(let e=0;e<l;++e)g(this.separator),g(this.rightPad)}else{for(let e=0;e<l-1;++e)g(this.rightPad),g(this.separator);g(this.rightPad)}}}compute(e,t){let n=e.length,r=t.length;if(r>0){let e=t[0];if(e!==0)throw Error(`First split value must be 0, got ${e}`);for(let i=1;i<r;++i){let r=t[i]>=e;if(r&&=t[i]<=n,!r)throw Error(`Invalid split value ${t[i]}, must be in [${e}, ${n}]`);e=t[i]}if(e!==n)throw Error(`Last split value must be data size. Expected ${n}, got ${e}`)}let i=r-1,a=E(`int32`,r);if(n===0||r===0){let e=Array(n);for(let e=0;e<=i;++e)a[e]=0;return[e,a]}a[0]=0;for(let e=1;e<=i;++e){let n=t[e]-t[e-1],r=0;this.nGramWidths.forEach(e=>{r+=this.getNumNGrams(n,e)}),this.preserveShort&&n>0&&r===0&&(r=1),a[e]=a[e-1]+r}let o=Array(a[i]);for(let n=0;n<i;++n){let r=t[n],i=a[n];if(this.nGramWidths.forEach(a=>{let s=t[n+1]-t[n],c=this.getNumNGrams(s,a);this.createNGrams(e,r,o,i,c,a),i+=c}),this.preserveShort&&i===a[n]){let a=t[n+1]-t[n];if(a===0)continue;let s=a+2*this.padWidth;this.createNGrams(e,r,o,i,1,s)}}return[o,a]}};function mb(e,t,n,r,i,a,o,s){return new pb(n,r,i,a,o,s).compute(e,t)}function hb(e,t,n,r){if(!e.length)return;if(t.length===0){for(let t=0;t<e.length;++t)r.push(e.subarray(t,t+1));return}if(t.length===1){let i=t[0],a=e.indexOf(i);for(;a!==-1;){let t=e.subarray(0,a);n&&t.length===0||r.push(t),e=e.subarray(a+1),a=e.indexOf(i)}n&&e.length===0||r.push(e);return}let i=0;for(let a=0;a<e.length+1;a++)if(a===e.length||t.indexOf(e[a])!==-1){let t=e.subarray(i,a);n&&t.length===0||r.push(t),i=a+1}}function gb(e,t,n){let r=e.length,i=[],a=0,o=0,s=Array(r);for(let c=0;c<r;++c){let r=i.length;hb(e[c],t,n,i);let l=i.length-r;s[c]=l,a+=l,o=Math.max(o,l)}let c=E(`int32`,a*2),l=Array(a),u=[r,o],d=0;for(let e=0;e<r;++e)for(let t=0;t<s[e];++t)c[d*2]=e,c[d*2+1]=t,l[d]=i[d],++d;return[c,l,u]}function _b(e,t){let n=E(`int32`,e.length);for(let r=0;r<e.length;++r)n[r]=Br(e[r]).modulo(t).getLowBitsUnsigned();return n}var vb=Tv(((e,t)=>e-t)),yb=Iv(`Sub`,vb,Lv(((e,t,n,r)=>({real:e-n,imag:t-r})))),bb={kernelName:`Sub`,backendName:`cpu`,kernelFunc:yb};function xb(e,t){let n=Array(e.rank);for(let r=0;r<n.length;r++)n[r]=e.shape[r]*t[r];let r=L(n,e.dtype);for(let t=0;t<r.values.length;++t){let n=r.indexToLoc(t),i=Array(e.rank);for(let t=0;t<i.length;t++)i[t]=n[t]%e.shape[t];let a=e.locToIndex(i);r.values[t]=e.values[a]}return r}var Sb=(e,t)=>{let n=t.value-e.value;return n===0?e.index-t.index:n};function Cb(e,t,n=0,r=e.length-1){for(;r>n;){if(r-n>600){let i=r-n+1,a=t-n+1,o=Math.log(i),s=.5*Math.exp(2*o/3),c=.5*Math.sqrt(o*s*(i-s)/i)*Math.sign(a-i/2);Cb(e,t,Math.max(n,Math.floor(t-a*s/i+c)),Math.min(r,Math.floor(t+(i-a)*s/i+c)))}let i=e[t],a=n,o=r;for(u(e,n,t),Sb(e[r],i)>0&&u(e,n,r);a<o;){for(u(e,a,o),a++,o--;Sb(e[a],i)<0;)a+=1;for(;Sb(e[o],i)>0;)--o}Sb(e[n],i)===0?u(e,n,o):(o+=1,u(e,o,r)),o<=t&&(n=o+1),t<=o&&(r=o-1)}}function wb(e,t,n,r,i){let a=t[t.length-1],[o,s]=[e.length/a,a],c=T(n,o*r),l=T(`int32`,o*r);for(let t=0;t<o;t++){let n=t*s,a=e.subarray(n,n+s),o=Array(a.length);a.forEach((e,t)=>o[t]={value:e,index:t}),r<o.length&&(Cb(o,r),o=o.slice(0,r)),i&&o.sort(Sb);let u=t*r,d=c.subarray(u,u+r),f=l.subarray(u,u+r);for(let e=0;e<r;e++)d[e]=o[e].value,f[e]=o[e].index}let u=t.slice();return u[u.length-1]=r,[L(u,n,c),L(u,`int32`,l)]}function Tb(e,t,n,r){let i=C(t,n)[0],a=[1,n[0],1];for(let e=0;e<i;e++)a[0]*=n[e];a[1]=n[i];for(let e=i+1;e<n.length;e++)a[2]*=n[e];let o={},s=new Int32Array(n[i]),c=new si(a,r,e),l=[],u=a[0]===1&&a[2]===1;for(let t=0;t<n[i];t++){let n;if(u)n=e[t].toString();else{let e=[];for(let n=0;n<a[0];n++)for(let r=0;r<a[2];r++)e.push(c.get(n,t,r));n=e.join(`,`)}if(o[n]!==void 0)s[t]=o[n];else{let e=Object.keys(o).length;o[n]=e,s[t]=e,l.push(t)}}let d=a.slice();d[1]=Object.keys(o).length;let f=new si(d,r);l.forEach((e,t)=>{for(let n=0;n<a[0];n++)for(let r=0;r<a[2];r++)f.set(c.get(n,e,r),n,t,r)});let p=n.slice();return p[i]=d[1],{outputValues:f.values,outputShape:p,indices:s}}var Eb=t({addImpl:()=>Rv,bincountImpl:()=>Vv,bincountReduceImpl:()=>Hv,castImpl:()=>Nv,ceilImpl:()=>Gv,concatImpl:()=>qv,equalImpl:()=>Jv,expImpl:()=>Zv,expm1Impl:()=>ey,floorImpl:()=>ny,gatherNdImpl:()=>iy,gatherV2Impl:()=>ay,greaterEqualImpl:()=>cy,greaterImpl:()=>oy,lessEqualImpl:()=>fy,lessImpl:()=>uy,linSpaceImpl:()=>my,logImpl:()=>hy,maxImpl:()=>_y,maximumImpl:()=>vy,minimumImpl:()=>by,multiplyImpl:()=>Sy,negImpl:()=>Ty,notEqualImpl:()=>Oy,prodImpl:()=>Ny,raggedGatherImpl:()=>Uy,raggedTensorToTensorImpl:()=>Jy,rangeImpl:()=>Yy,rsqrtImpl:()=>Xy,scatterImpl:()=>Qy,sigmoidImpl:()=>$y,simpleAbsImpl:()=>Cv,sliceImpl:()=>nb,sparseFillEmptyRowsImpl:()=>ab,sparseReshapeImpl:()=>ob,sparseSegmentReductionImpl:()=>sb,sqrtImpl:()=>cb,squaredDifferenceImpl:()=>ub,stridedSliceImpl:()=>fb,stringNGramsImpl:()=>mb,stringSplitImpl:()=>gb,stringToHashBucketFastImpl:()=>_b,subImpl:()=>vb,tileImpl:()=>xb,topKImpl:()=>wb,transposeImpl:()=>Ay,uniqueImpl:()=>Tb});Jo(`cpu`,()=>new Sv,1);var Db=K(`Elu`,e=>e>=0?e:Math.exp(e)-1),Ob={kernelName:`Elu`,backendName:`cpu`,kernelFunc:Db};function kb(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{alpha:a}=r;G([i],`leakyRelu`);let o=g(i.shape),s=n.data.get(i.dataId).values,c=T(`float32`,o);for(let e=0;e<s.length;e++)c[e]=s[e]<0?a*s[e]:s[e];return n.makeTensorInfo(i.shape,`float32`,c)}var Ab={kernelName:Lt,backendName:`cpu`,kernelFunc:kb},jb=Tv((e,t)=>e<0?t*e:e);function Mb(e){let{inputs:t,backend:n}=e,{x:r,alpha:i}=t;G([r,i],`prelu`);let a=n.data.get(r.dataId).values,o=n.data.get(i.dataId).values,[s,c]=jb(r.shape,i.shape,a,o,`float32`);return n.makeTensorInfo(c,`float32`,s)}var Nb={kernelName:fn,backendName:`cpu`,kernelFunc:Mb},Pb=K(yn,e=>Math.max(0,e)),Fb={kernelName:yn,backendName:`cpu`,kernelFunc:Pb},Ib=K(Tn,e=>Math.min(Math.max(0,e),6)),Lb={kernelName:Tn,backendName:`cpu`,kernelFunc:Ib};function Rb(e,t,n,r,i){if(n===`linear`)return kv({inputs:{x:t},backend:e});if(n===`relu`)return Pb({inputs:{x:t},backend:e});if(n===`elu`)return Db({inputs:{x:t},backend:e});if(n===`relu6`)return Ib({inputs:{x:t},backend:e});if(n===`prelu`)return Mb({inputs:{x:t,alpha:r},backend:e});if(n===`leakyrelu`)return kb({inputs:{x:t},backend:e,attrs:{alpha:i}});if(n===`sigmoid`)return eb({inputs:{x:t},backend:e});throw Error(`Activation ${n} has not been implemented for the CPU backend.`)}function q(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{shape:a}=r,o=g(i.shape),s=S(a,o),c=g(s);f(o===c,()=>`The new shape (${s}) has ${c} elements and the old shape (${i.shape}) has ${o} elements. The new shape and old shape must have the same number of elements.`),n.incRef(i.dataId);let l=n.data.get(i.dataId);if(l.complexTensorInfos!=null){let e=l.complexTensorInfos.real,t=l.complexTensorInfos.imag;e.shape=s,t.shape=s}return{dataId:i.dataId,shape:s,dtype:i.dtype}}var zb={kernelName:bn,backendName:`cpu`,kernelFunc:q};function Bb(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a}=t,{transposeA:o,transposeB:s}=r;G([i,a],`matMul`);let c=i.shape.length,l=a.shape.length,u=o?i.shape[c-2]:i.shape[c-1],d=s?a.shape[l-1]:a.shape[l-2],p=o?i.shape[c-1]:i.shape[c-2],m=s?a.shape[l-2]:a.shape[l-1],h=i.shape.slice(0,-2),_=a.shape.slice(0,-2),v=g(h),y=g(_),b=z(i.shape.slice(0,-2),a.shape.slice(0,-2)).concat([p,m]);f(u===d,()=>`Error in matMul: inner shapes (${u}) and (${d}) of Tensors with shapes ${i.shape} and ${a.shape} and transposeA=${o} and transposeB=${s} must match.`);let x=o?[v,u,p]:[v,p,u],S=s?[y,m,d]:[y,d,m],C=q({inputs:{x:i},backend:n,attrs:{shape:x}}),w=q({inputs:{x:a},backend:n,attrs:{shape:S}}),T=o?C.shape[1]:C.shape[2],E=o?C.shape[2]:C.shape[1],D=s?w.shape[1]:w.shape[2],O=Math.max(v,y),ee=n.data.get(C.dataId).values,k=n.data.get(w.dataId).values,A=j(C.shape),te=j(w.shape),[ne,re,ie]=o?[A[0],1,A[1]]:[A[0],A[1],1],[ae,oe,se]=s?[1,te[1],te[0]]:[te[1],1,te[0]],ce=E*D,le=L([O,E,D],C.dtype),ue=le.values,M=n.blockSize;for(let e=0;e<O;e++)for(let t=0;t<E;t+=M)for(let n=0;n<D;n+=M)for(let r=0;r<T;r+=M){let i=Math.min(t+M,E),a=Math.min(n+M,D),o=Math.min(r+M,T);for(let s=t;s<i;s++)for(let t=n;t<a;t++){let n=0;for(let i=r;i<o;i++){let r=Math.min(e,v-1)*ne,a=Math.min(e,y-1)*se,o=ee[r+s*re+i*ie],c=k[i*ae+t*oe+a];n+=o*c}ue[e*ce+(s*D+t)]+=n}}return n.disposeIntermediateTensorInfo(C),n.disposeIntermediateTensorInfo(w),n.makeTensorInfo(b,le.dtype,le.values)}var Vb={kernelName:Ve,backendName:`cpu`,kernelFunc:Bb};function Hb(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a,bias:o,preluActivationWeights:s}=t,{transposeA:c,transposeB:l,activation:u,leakyreluAlpha:d}=r,f,p,m,h=[];f=Bb({inputs:{a:i,b:a},attrs:{transposeA:c,transposeB:l},backend:n}),o&&(p=zv({inputs:{a:f,b:o},backend:n}),h.push(f),f=p),u&&(m=Rb(n,f,u,s,d),h.push(f),f=m);for(let e of h)n.disposeIntermediateTensorInfo(e);return f}var Ub={kernelName:dr,backendName:`cpu`,kernelFunc:Hb},Wb={kernelName:De,backendName:`cpu`,kernelFunc:K(De,e=>Math.acos(e))},Gb={kernelName:Oe,backendName:`cpu`,kernelFunc:K(Oe,e=>Math.acosh(e))};function Kb(e){let{inputs:t,backend:n}=e,r=t;G(t,`addN`);let i=r.map(e=>n.data.get(e.dataId).values),a=L(r[0].shape,r[0].dtype),o=a.values;for(let e=0;e<r.length;e++){let t=i[e];for(let e=0;e<o.length;e++)o[e]+=t[e]}return n.makeTensorInfo(a.shape,a.dtype,a.values)}var qb={kernelName:ke,backendName:`cpu`,kernelFunc:Kb};function Jb(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;G(i,`all`);let s=C(a,i.shape),c=s,l=_u(c,i.shape.length),u=i;l!=null&&(u=jy({inputs:{x:i},backend:n,attrs:{perm:l}}),c=yu(c.length,i.shape.length)),gu(`all`,c,u.shape.length);let[d,f]=mu(u.shape,c),p=g(f),m=M(g(d),u.dtype),h=n.data.get(u.dataId).values;for(let e=0;e<m.length;++e){let t=e*p,n=h[t];for(let e=0;e<p;++e){let r=h[t+e];n&&=r}m[e]=n}l!=null&&n.disposeIntermediateTensorInfo(u);let _=n.makeTensorInfo(d,u.dtype,m);if(o){let e=hu(d,s),t=q({inputs:{x:_},backend:n,attrs:{shape:e}});return n.disposeIntermediateTensorInfo(_),t}return _}var Yb={kernelName:`All`,backendName:`cpu`,kernelFunc:Jb};function Xb(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;G(i,`any`);let s=C(a,i.shape),c=s,l=_u(c,i.shape.length),u=i;l!=null&&(u=jy({inputs:{x:i},backend:n,attrs:{perm:l}}),c=yu(c.length,i.shape.length)),gu(`any`,c,u.shape.length);let[d,f]=mu(u.shape,c),p=g(f),m=M(g(d),u.dtype),h=n.data.get(u.dataId).values;for(let e=0;e<m.length;++e){let t=e*p,n=h[t];for(let e=0;e<p;++e){let r=h[t+e];n||=r}m[e]=n}l!=null&&n.disposeIntermediateTensorInfo(u);let _=n.makeTensorInfo(d,u.dtype,m);if(o){let e=hu(d,s),t=q({inputs:{x:_},backend:n,attrs:{shape:e}});return n.disposeIntermediateTensorInfo(_),t}return _}var Zb={kernelName:`Any`,backendName:`cpu`,kernelFunc:Xb};function Qb(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a}=r;G(i,`argMax`);let o=C(a,i.shape),s=_u(o,i.shape.length),c=i,l=[];s!=null&&(c=jy({inputs:{x:i},backend:n,attrs:{perm:s}}),l.push(c),o=yu(o.length,c.shape.length)),o=[o[0]],gu(`argMax`,o,c.shape.length);let[u,d]=mu(c.shape,o),f=M(g(u),`int32`),p=g(d),m=n.data.get(c.dataId).values;for(let e=0;e<f.length;++e){let t=e*p,n=m[t],r=0;for(let e=0;e<p;++e){let i=m[t+e];i>n&&(n=i,r=e)}f[e]=r}return l.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(u,`int32`,f)}var $b={kernelName:Ae,backendName:`cpu`,kernelFunc:Qb};function ex(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a}=r;G(i,`argMin`);let o=C(a,i.shape),s=_u(o,i.shape.length),c=i,l=[];s!=null&&(c=jy({inputs:{x:i},backend:n,attrs:{perm:s}}),l.push(c),o=yu(o.length,c.shape.length)),o=[o[0]],gu(`argMin`,o,c.shape.length);let[u,d]=mu(c.shape,o),f=M(g(u),`int32`),p=g(d),m=n.data.get(c.dataId).values;for(let e=0;e<f.length;++e){let t=e*p,n=m[t],r=0;for(let e=0;e<p;++e){let i=m[t+e];i<n&&(n=i,r=e)}f[e]=r}return l.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(u,`int32`,f)}var tx={kernelName:je,backendName:`cpu`,kernelFunc:ex},nx={kernelName:Me,backendName:`cpu`,kernelFunc:K(Me,e=>Math.asin(e))},rx={kernelName:Ne,backendName:`cpu`,kernelFunc:K(Ne,e=>Math.asinh(e))},ix={kernelName:Pe,backendName:`cpu`,kernelFunc:K(Pe,e=>Math.atan(e))},ax={kernelName:Ie,backendName:`cpu`,kernelFunc:Iv(Ie,Tv((e,t)=>Math.atan2(e,t)))},ox={kernelName:Fe,backendName:`cpu`,kernelFunc:K(Fe,e=>Math.atanh(e))};function sx(e,t,n,r,i,a){let o=i.strideHeight,s=i.strideWidth,c=i.dilationHeight,l=i.dilationWidth,u=i.effectiveFilterHeight,d=i.effectiveFilterWidth,f=i.padInfo.top,p=i.padInfo.left,m=a===`max`?-1/0:1/0,h=L(i.outShape,n),g=h.values,_=i.outShape[1]*i.outShape[2]*i.outShape[3],v=i.outShape[2]*i.outShape[3],y=i.outShape[3];for(let t=0;t<i.batchSize;++t){let n=t*_,h=t*r[0];for(let t=0;t<i.inChannels;++t)for(let _=0;_<i.outHeight;++_){let b=_*o-f,x=Math.max(0,b),S=Math.min(i.inHeight,u+b),C=n+_*v;for(let n=0;n<i.outWidth;++n){let o=n*s-p,u=Math.max(0,o),f=Math.min(i.inWidth,d+o),_=m,v=0,b=0;for(let n=x;n<S;n+=c){let i=h+n*r[1];for(let n=u;n<f;n+=l){let o=e[i+n*r[2]+t];a===`max`&&o>_?_=o:a===`avg`&&(v+=o,b++)}if(isNaN(_))break}let w=C+n*y+t;g[w]=a===`avg`?v/b:_}}}return h}function cx(e,t,n,r,i=!1,a=!1){let o=L(r.outShape,`int32`),s=r.strideHeight,c=r.strideWidth,l=r.dilationHeight,u=r.dilationWidth,d=r.effectiveFilterHeight,f=r.effectiveFilterWidth,p=r.padInfo.top,m=r.padInfo.left,h=L(t,n,e);for(let e=0;e<r.batchSize;++e)for(let t=0;t<r.inChannels;++t)for(let n=0;n<r.outHeight;++n){let g=n*s-p,_=g;for(;_<0;)_+=l;let v=Math.min(r.inHeight,d+g);for(let s=0;s<r.outWidth;++s){let d=s*c-m,p=d;for(;p<0;)p+=u;let y=Math.min(r.inWidth,f+d),b=-1/0,x=-1;for(let n=_;n<v;n+=l){let o=n-g;for(let s=p;s<y;s+=u){let c=s-d,l=h.get(e,n,s,t);l>b&&(b=l,x=i?a?((e*r.inHeight+n)*r.inWidth+s)*r.inChannels+t:(n*r.inWidth+s)*r.inChannels+t:o*f+c)}}o.set(x,e,n,s,t)}}return o}function lx(e,t,n,r,i,a){let o=i.strideDepth,s=i.strideHeight,c=i.strideWidth,l=i.dilationDepth,u=i.dilationHeight,d=i.dilationWidth,f=i.effectiveFilterDepth,p=i.effectiveFilterHeight,m=i.effectiveFilterWidth,h=i.padInfo.front,g=i.padInfo.top,_=i.padInfo.left,v=a===`max`?-1/0:1/0,y=L(i.outShape,n),b=y.values,x=i.outShape[1]*i.outShape[2]*i.outShape[3]*i.outShape[4],S=i.outShape[2]*i.outShape[3]*i.outShape[4],C=i.outShape[3]*i.outShape[4],w=i.outShape[4];for(let t=0;t<i.batchSize;++t){let n=t*x,y=t*r[0];for(let t=0;t<i.inChannels;++t)for(let x=0;x<i.outDepth;++x){let T=x*o-h,E=T;for(;E<0;)E+=l;let D=Math.min(i.inDepth,f+T),O=n+x*S;for(let n=0;n<i.outHeight;++n){let o=n*s-g,f=o;for(;f<0;)f+=u;let h=Math.min(i.inHeight,p+o),x=O+n*C;for(let n=0;n<i.outWidth;++n){let o=n*c-_,s=o;for(;s<0;)s+=d;let p=Math.min(i.inWidth,m+o),g=x+n*w,S=v,C=0,T=0;for(let n=E;n<D;n+=l){let i=y+n*r[1];for(let n=f;n<h;n+=u){let o=i+n*r[2];for(let n=s;n<p;n+=d){let i=e[o+n*r[3]+t];if(a===`max`&&i>S?S=i:a===`avg`&&(C+=i,T++),isNaN(S))break}if(isNaN(S))break}if(isNaN(S))break}let O=g+t;b[O]=a===`avg`?C/T:S}}}}return y}function ux(e,t){let n=L(t.outShape,`int32`),r=t.strideDepth,i=t.strideHeight,a=t.strideWidth,o=t.dilationDepth,s=t.dilationHeight,c=t.dilationWidth,l=t.effectiveFilterDepth,u=t.effectiveFilterHeight,d=t.effectiveFilterWidth,f=t.padInfo.front,p=t.padInfo.top,m=t.padInfo.left;for(let h=0;h<t.batchSize;++h)for(let g=0;g<t.inChannels;++g)for(let _=0;_<t.outDepth;++_){let v=_*r-f,y=v;for(;y<0;)y+=o;let b=Math.min(t.inDepth,l+v);for(let r=0;r<t.outHeight;++r){let l=r*i-p,f=l;for(;f<0;)f+=s;let x=Math.min(t.inHeight,u+l);for(let i=0;i<t.outWidth;++i){let p=i*a-m,S=p;for(;S<0;)S+=c;let C=Math.min(t.inWidth,d+p),w=-1/0,T=-1;for(let t=y;t<b;t+=o){let n=t-v;for(let r=f;r<x;r+=s){let i=r-l;for(let a=S;a<C;a+=c){let o=a-p,s=e.get(h,t,r,a,g);s>=w&&(w=s,T=n*u*d+i*u+o)}}}n.set(T,h,_,r,i,g)}}}return n}function dx(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t;G(i,`avgPool`);let{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=r;f(Tc(o,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${o} and dilations '1'`);let l=dc(i.shape,a,o,1,s,c),u;if(l.filterWidth===1&&l.filterHeight===1&&_(l.inShape,l.outShape))u=kv({inputs:{x:i},backend:n});else{let e=n.data.get(i.dataId).values,t=j(i.shape),r=sx(e,i.shape,i.dtype,t,l,`avg`);u=n.makeTensorInfo(l.outShape,i.dtype,r.values)}return u}var fx={kernelName:Le,backendName:`cpu`,kernelFunc:dx};function px(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c,dataFormat:l}=r;G(i,`avgPool3d`);let u=fc(i.shape,a,o,1,s,c,l),d=n.data.get(i.dataId).values,f=lx(d,i.shape,i.dtype,j(i.shape),u,`avg`);return n.makeTensorInfo(f.shape,`float32`,f.values)}var mx={kernelName:ze,backendName:`cpu`,kernelFunc:px};function hx(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=r;G([i,a],`avgPool3DGrad`);let u=fc(a.shape,o,s,1,c,l),d=u.strideDepth,f=u.strideHeight,p=u.strideWidth,m=u.filterDepth,h=u.filterHeight,g=u.filterWidth,_=u.dilationDepth,v=u.dilationHeight,y=u.dilationWidth,b=u.effectiveFilterDepth,x=u.effectiveFilterHeight,S=u.effectiveFilterWidth,C=b-1-u.padInfo.front,w=S-1-u.padInfo.left,T=x-1-u.padInfo.top,E=L(a.shape,`float32`),D=1/(m*h*g),O=n.bufferSync(i);for(let e=0;e<u.batchSize;++e)for(let t=0;t<u.inChannels;++t)for(let n=0;n<u.inDepth;++n)for(let r=0;r<u.inHeight;++r)for(let i=0;i<u.inWidth;++i){let a=n-C,o=r-T,s=i-w,c=0;for(let n=0;n<b;n+=_){let r=(a+n)/d;if(!(r<0||r>=u.outDepth||Math.floor(r)!==r))for(let n=0;n<x;n+=v){let i=(o+n)/f;if(!(i<0||i>=u.outHeight||Math.floor(i)!==i))for(let n=0;n<S;n+=y){let a=(s+n)/p;if(a<0||a>=u.outWidth||Math.floor(a)!==a)continue;let o=O.get(e,r,i,a,t);c+=o}}}E.set(c*D,e,n,r,i,t)}return n.makeTensorInfo(E.shape,E.dtype,E.values)}var gx={kernelName:Be,backendName:`cpu`,kernelFunc:hx};function _x(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a;G([i,a],`avgPoolGrad`);let{filterSize:s,strides:c,pad:l}=r,u=dc(o.shape,s,c,1,l),d=u.strideHeight,f=u.strideWidth,p=u.filterHeight,m=u.filterWidth,h=u.dilationHeight,g=u.dilationWidth,_=u.effectiveFilterHeight,v=u.effectiveFilterWidth,y=v-1-u.padInfo.left,b=_-1-u.padInfo.top,x=L(o.shape,`float32`),S=1/(p*m),C=n.data.get(i.dataId).values,w=L(i.shape,`float32`,C);for(let e=0;e<u.batchSize;++e)for(let t=0;t<u.inChannels;++t)for(let n=0;n<u.inHeight;++n)for(let r=0;r<u.inWidth;++r){let i=n-b,a=r-y,o=0;for(let n=0;n<_;n+=h){let r=(i+n)/d;if(!(r<0||r>=u.outHeight||Math.floor(r)!==r))for(let n=0;n<v;n+=g){let i=(a+n)/f;if(i<0||i>=u.outWidth||Math.floor(i)!==i)continue;let s=w.get(e,r,i,t);o+=s}}x.set(o*S,e,n,r,t)}return n.makeTensorInfo(x.shape,x.dtype,x.values)}var vx={kernelName:Re,backendName:`cpu`,kernelFunc:_x};function yx(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,scale:a,offset:o,mean:s,variance:c}=t;f(s.shape.length===c.shape.length,()=>`Batch normalization gradient requires mean and variance to have equal ranks.`),f(o==null||s.shape.length===o.shape.length,()=>`Batch normalization gradient requires mean and offset to have equal ranks.`),f(a==null||s.shape.length===a.shape.length,()=>`Batch normalization gradient requires mean and scale to have equal ranks.`),G([i,s,c,a,o],`batchNorm`);let{varianceEpsilon:l}=r;l??=.001;let u=n.data.get(i.dataId).values,d=n.data.get(s.dataId).values,p=n.data.get(c.dataId).values,m=a?n.data.get(a.dataId).values:new Float32Array([1]),h=o?n.data.get(o.dataId).values:new Float32Array([0]),g=new Float32Array(u.length),_=h.length,v=m.length,y=p.length,b=d.length,x=0,S=0,C=0,w=0;for(let e=0;e<u.length;++e)g[e]=h[x++]+(u[e]-d[S++])*m[C++]/Math.sqrt(p[w++]+l),x>=_&&(x=0),S>=b&&(S=0),C>=v&&(C=0),w>=y&&(w=0);return n.makeTensorInfo(i.shape,i.dtype,g)}var bx={kernelName:Et,backendName:`cpu`,kernelFunc:yx};function xx(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,crops:o}=r;G([i],`batchToSpaceND`);let s=a.reduce((e,t)=>e*t),c=w_(i.shape,a,s),l=T_(c.length,a.length),u=E_(i.shape,a,s),d=D_(o,a.length),f=O_(u,o,a.length),p=q({inputs:{x:i},backend:n,attrs:{shape:c}}),m=jy({inputs:{x:p},backend:n,attrs:{perm:l}}),h=q({inputs:{x:m},backend:n,attrs:{shape:u}}),g=rb({inputs:{x:h},backend:n,attrs:{begin:d,size:f}});return n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(h),g}var Sx={kernelName:He,backendName:`cpu`,kernelFunc:xx};function Cx(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,weights:a}=t,{size:o}=r,s=n.data.get(i.dataId).values,c=n.data.get(a.dataId).values,l=Vv(s,c,a.dtype,a.shape,o);return n.makeTensorInfo([o],a.dtype,l)}var wx={kernelName:Ue,backendName:`cpu`,kernelFunc:Cx};function Tx(e){let{inputs:t,backend:n}=e,{s0:r,s1:i}=t,a=n.data.get(r.dataId).values,o=n.data.get(i.dataId).values,s=z(Array.from(a),Array.from(o));return n.makeTensorInfo([s.length],`int32`,Int32Array.from(s))}var Ex={kernelName:We,backendName:`cpu`,kernelFunc:Tx},Dx={kernelName:qe,backendName:`cpu`,kernelFunc:K(qe,(e,t)=>{let n=t;return e>n.clipValueMax?n.clipValueMax:e<n.clipValueMin?n.clipValueMin:e})},Ox={kernelName:Ye,backendName:`cpu`,kernelFunc:e=>{let{x:t}=e.inputs,n=e.backend,r=new Float32Array(g(t.shape)),i=n.data.get(t.dataId),a=i.complexTensorInfos.real,o=i.complexTensorInfos.imag,s=n.data.get(a.dataId).values,c=n.data.get(o.dataId).values;for(let e=0;e<s.length;e++){let t=s[e],n=c[e];r[e]=Math.hypot(t,n)}return n.makeOutput(r,t.shape,`float32`)}};function kx(e){let{inputs:t,backend:n}=e,{input:r}=t,i=n.data.get(r.dataId).complexTensorInfos.imag,a=n.data.get(i.dataId).values;return n.makeTensorInfo(i.shape,i.dtype,a)}var Ax={kernelName:Nt,backendName:`cpu`,kernelFunc:kx};function jx(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r,a=C(i,t[0].shape)[0];h_(t.map(e=>e.shape),a);let o=g_(t.map(e=>e.shape),a);if(g(o)===0)return n.makeTensorInfo(o,t[0].dtype,[]);let s=t.filter(e=>g(e.shape)>0);if(s.length===1)return kv({inputs:{x:s[0]},backend:n});if(s[0].dtype===`complex64`){let e=s.map(e=>jv({inputs:{input:e},backend:n})),t=s.map(e=>kx({inputs:{input:e},backend:n})),r=jx({inputs:e,backend:n,attrs:{axis:a}}),i=jx({inputs:t,backend:n,attrs:{axis:a}}),o=Ev({inputs:{real:r,imag:i},backend:n});return e.forEach(e=>n.disposeIntermediateTensorInfo(e)),t.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.disposeIntermediateTensorInfo(r),n.disposeIntermediateTensorInfo(i),o}let c=s.map(e=>{let t=[-1,g(e.shape.slice(a))];return q({inputs:{x:e},backend:n,attrs:{shape:t}})}),l=c.map(e=>({vals:n.data.get(e.dataId).values,shape:e.shape}));o=g_(c.map(e=>e.shape),1);let u=c[0].shape[0]===1,d=qv(l,o,t[0].dtype,u),f=g_(s.map(e=>e.shape),a),p=n.makeTensorInfo(f,t[0].dtype,d);return c.forEach(e=>n.disposeIntermediateTensorInfo(e)),p}var Mx={kernelName:Xe,backendName:`cpu`,kernelFunc:jx};function Nx(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dataFormat:c,dilations:l,dimRoundingMode:u}=r;G([i,a],`conv2d`);let d=Ec(c),f=pc(i.shape,a.shape,o,l,s,u,!1,d),p=f.filterHeight,m=f.filterWidth,h=f.dilationHeight,g=f.dilationWidth,_=f.padInfo.left,v=f.padInfo.top,y=f.dataFormat===`channelsLast`,b=new si(f.outShape,i.dtype),x=j(i.shape),S=j(a.shape),C=x[0],w=y?x[1]:x[2],T=y?x[2]:1,E=y?1:x[1],D=b.strides[0],O=y?b.strides[1]:b.strides[2],ee=y?b.strides[2]:1,k=y?1:b.strides[1],A=n.data.get(i.dataId).values,te=n.data.get(a.dataId).values,ne=b.values;for(let e=0;e<f.batchSize;++e){let t=e*C,n=e*D;for(let e=0;e<f.outHeight;++e){let r=n+e*O,i=e*f.strideHeight-v;for(let e=0;e<p;++e){let n=i+e*h;if(n<0||n>=f.inHeight)continue;let a=e*S[0],o=t+n*w;for(let e=0;e<f.outWidth;++e){let t=r+e*ee,n=e*f.strideWidth-_;for(let e=0;e<m;++e){let r=n+e*g;if(r<0||r>=f.inWidth)continue;let i=a+e*S[1],s=o+r*T,c=i;for(let e=0;e<f.inChannels;++e){let n=A[s+e*E];for(let e=0;e<f.outChannels;++e)ne[t+e*k]+=n*te[c+e];c+=f.outChannels}}}}}}return n.makeTensorInfo(b.shape,b.dtype,ne)}var Px={kernelName:Ze,backendName:`cpu`,kernelFunc:Nx};function Fx(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,dataFormat:c,dimRoundingMode:l,filterShape:u}=r;G([i,a],`conv2dBackpropFilter`);let d=Ec(c),f=pc(i.shape,u,o,1,s,l,!1,d),{strideHeight:p,strideWidth:m,filterHeight:h,filterWidth:g}=f,_=f.dataFormat===`channelsLast`,v=new si(f.filterShape,`float32`),y=f.padInfo.left,b=f.padInfo.top,x=n.data.get(i.dataId).values,S=n.data.get(a.dataId).values,C=new si(i.shape,i.dtype,x),w=new si(a.shape,a.dtype,S);for(let e=0;e<h;++e){let t=Math.max(0,Math.ceil((b-e)/p)),n=Math.min(f.outHeight,(f.inHeight+b-e)/p);for(let r=0;r<g;++r){let i=Math.max(0,Math.ceil((y-r)/m)),a=Math.min(f.outWidth,(f.inWidth+y-r)/m);for(let o=0;o<f.inChannels;++o)for(let s=0;s<f.outChannels;++s){let c=0;for(let l=0;l<f.batchSize;++l)for(let u=t;u<n;++u){let t=e+u*p-b;for(let e=i;e<a;++e){let n=r+e*m-y;c+=_?C.get(l,t,n,o)*w.get(l,u,e,s):C.get(l,o,t,n)*w.get(l,s,u,e)}}v.set(c,e,r,o,s)}}}return n.makeTensorInfo(v.shape,v.dtype,v.values)}var Ix={kernelName:Qe,backendName:`cpu`,kernelFunc:Fx};function Lx(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{inputShape:o,strides:s,pad:c,dataFormat:l,dimRoundingMode:u}=r;G([i,a],`conv2dBackpropInput`);let d=j(a.shape),f=j(i.shape),p=Ec(l),m=pc(o,a.shape,s,1,c,u,!1,p),h=new si(m.inShape,`float32`),g=h.values,_=n.data.get(i.dataId).values,v=n.data.get(a.dataId).values,[y,b,x]=d,{batchSize:S,filterHeight:C,filterWidth:w,inChannels:T,inHeight:E,inWidth:D,outChannels:O,outHeight:ee,outWidth:k,strideHeight:A,strideWidth:te}=m;p=m.dataFormat;let ne=C-1-m.padInfo.top,re=w-1-m.padInfo.left,ie=p===`channelsLast`,ae=h.strides[0],oe=ie?h.strides[1]:h.strides[2],se=ie?h.strides[2]:1,ce=ie?1:h.strides[1],le=f[0],ue=ie?f[1]:f[2],M=ie?f[2]:1,de=ie?1:f[1];for(let e=0;e<S;++e)for(let t=0;t<T;++t)for(let n=0;n<E;++n){let r=n-ne,i=Math.max(0,Math.ceil(r/A)),a=Math.min(ee,(C+r)/A);for(let o=0;o<D;++o){let s=o-re,c=Math.max(0,Math.ceil(s/te)),l=Math.min(k,(w+s)/te),u=0;for(let n=i;n<a;++n){let i=n*A-r;for(let r=c;r<l;++r){let a=r*te-s,o=le*e+ue*n+M*r,c=y*(C-1-i)+b*(w-1-a)+x*t;for(let e=0;e<O;++e){let t=_[o+de*e],n=v[c+e];u+=t*n}}}let d=ae*e+oe*n+se*o+ce*t;g[d]=u}}return n.makeTensorInfo(h.shape,h.dtype,h.values)}var Rx={kernelName:$e,backendName:`cpu`,kernelFunc:Lx};function zx(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r;G([i,a],`conv3d`);let l=mc(i.shape,a.shape,o,c,s),{filterDepth:u,filterHeight:d,filterWidth:f,dilationDepth:p,dilationHeight:m,dilationWidth:h,padInfo:g}=l,_=g.front,v=g.left,y=g.top,b=new si(l.outShape,i.dtype),x=n.data.get(i.dataId).values,S=n.data.get(a.dataId).values,C=b.values,w=j(i.shape),T=j(a.shape);for(let e=0;e<l.batchSize;++e){let t=e*w[0],n=e*b.strides[0];for(let e=0;e<l.outDepth;++e){let r=n+e*b.strides[1],i=e*l.strideDepth-_;for(let e=0;e<u;++e){let n=i+e*p;if(n<0||n>=l.inDepth)continue;let a=e*T[0],o=t+n*w[1];for(let e=0;e<l.outHeight;++e){let t=r+e*b.strides[2],n=e*l.strideHeight-y;for(let e=0;e<d;++e){let r=n+e*m;if(r<0||r>=l.inHeight)continue;let i=a+e*T[1],s=o+r*w[2];for(let e=0;e<l.outWidth;++e){let n=t+e*l.outChannels,r=e*l.strideWidth-v;for(let e=0;e<f;++e){let t=r+e*h;if(t<0||t>=l.inWidth)continue;let a=i+e*T[2],o=s+t*l.inChannels,c=a;for(let e=0;e<l.inChannels;++e){let t=x[o+e];for(let e=0;e<l.outChannels;++e)C[n+e]+=t*S[c+e];c+=l.outChannels}}}}}}}}return n.makeTensorInfo(b.shape,b.dtype,b.values)}var Bx={kernelName:et,backendName:`cpu`,kernelFunc:zx};function Vx(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,filterShape:c}=r;G([i,a],`conv3dBackpropFilterV2`);let l=j(i.shape),u=j(a.shape),d=mc(i.shape,c,o,1,s),f=d.strideDepth,p=d.strideHeight,m=d.strideWidth,h=d.filterDepth,g=d.filterHeight,_=d.filterWidth,v=new si(d.filterShape,`float32`),y=v.values,[b,x,S,C]=v.strides,w=n.data.get(a.dataId).values,[T,E,D,O]=u,ee=n.data.get(i.dataId).values,[k,A,te,ne]=l,re=d.padInfo.front,ie=d.padInfo.left,ae=d.padInfo.top;for(let e=0;e<h;++e){let t=Math.max(0,Math.ceil((re-e)/f)),n=Math.min(d.outDepth,(d.inDepth+re-e)/f),r=e*b;for(let i=0;i<g;++i){let a=Math.max(0,Math.ceil((ae-i)/p)),o=Math.min(d.outHeight,(d.inHeight+ae-i)/p),s=i*x+r;for(let r=0;r<_;++r){let c=Math.max(0,Math.ceil((ie-r)/m)),l=Math.min(d.outWidth,(d.inWidth+ie-r)/m),u=r*S+s;for(let s=0;s<d.inChannels;++s){let h=s*C+u;for(let u=0;u<d.outChannels;++u){let g=0;for(let h=0;h<d.batchSize;++h){let d=h*k,_=h*T;for(let h=t;h<n;++h){let t=(e+h*f-re)*A+d,n=h*E+_;for(let e=a;e<o;++e){let a=(i+e*p-ae)*te+t,o=e*D+n;for(let e=c;e<l;++e){let t=(r+e*m-ie)*ne+a,n=e*O+o;g+=ee[t+s]*w[n+u]}}}}y[h+u]=g}}}}}return n.makeTensorInfo(v.shape,v.dtype,v.values)}var Hx={kernelName:tt,backendName:`cpu`,kernelFunc:Vx};function Ux(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{pad:o,strides:s,inputShape:c}=r;G([i],`conv3dBackpropInputV2`);let l=j(i.shape),u=j(a.shape),d=mc(c,a.shape,s,1,o),f=new si(d.inShape,`float32`),p=f.values,[m,h,g,_]=f.strides,v=n.data.get(i.dataId).values,[y,b,x,S]=l,C=n.data.get(a.dataId).values,[w,T,E,D]=u,{batchSize:O,filterDepth:ee,filterHeight:k,filterWidth:A,inChannels:te,inDepth:ne,inHeight:re,inWidth:ie,outChannels:ae,outDepth:oe,outHeight:se,outWidth:ce,strideDepth:le,strideHeight:ue,strideWidth:M}=d,de=ee-1-d.padInfo.front,fe=k-1-d.padInfo.top,pe=A-1-d.padInfo.left;for(let e=0;e<O;++e)for(let t=0;t<te;++t)for(let n=0;n<ne;++n){let r=n-de,i=Math.max(0,Math.ceil(r/le)),a=Math.min(oe,(ee+r)/le);for(let o=0;o<re;++o){let s=o-fe,c=Math.max(0,Math.ceil(s/ue)),l=Math.min(se,(k+s)/ue);for(let u=0;u<ie;++u){let d=u-pe,f=Math.max(0,Math.ceil(d/M)),O=Math.min(ce,(A+d)/M),te=0;for(let n=i;n<a;++n){let i=n*le-r;for(let r=c;r<l;++r){let a=r*ue-s;for(let o=f;o<O;++o){let s=o*M-d,c=y*e+b*n+x*r+S*o,l=w*(ee-1-i)+T*(k-1-a)+E*(A-1-s)+D*t;for(let e=0;e<ae;++e){let t=v[c+e],n=C[l+e];te+=t*n}}}}p[m*e+h*n+g*o+_*u+t]=te}}}return n.makeTensorInfo(f.shape,f.dtype,f.values)}var Wx={kernelName:nt,backendName:`cpu`,kernelFunc:Ux},Gx={kernelName:`Cos`,backendName:`cpu`,kernelFunc:K(`Cos`,e=>Math.cos(e))},Kx={kernelName:rt,backendName:`cpu`,kernelFunc:K(rt,e=>Math.cosh(e))};function qx(e){let{inputs:t,backend:n,attrs:r}=e,{image:i,boxes:a,boxInd:o}=t,{cropSize:s,method:c,extrapolationValue:l}=r,[u,d,f,p]=i.shape,m=a.shape[0],[h,g]=s,_=L([m,h,g,p],`float32`),v=n.data.get(a.dataId).values,y=n.data.get(o.dataId).values,b=n.data.get(i.dataId).values,x=j(i.shape),S=j(_.shape);for(let e=0;e<m;e++){let t=e*4,n=v[t],r=v[t+1],i=v[t+2],a=v[t+3],o=y[e];if(o>=u)continue;let s=h>1?(i-n)*(d-1)/(h-1):0,m=g>1?(a-r)*(f-1)/(g-1):0;for(let t=0;t<h;t++){let u=h>1?n*(d-1)+t*s:.5*(n+i)*(d-1);if(u<0||u>d-1)for(let n=0;n<g;n++)for(let r=0;r<p;r++){let i=r+n*S[2]+t*S[1]+e*S[0];_.values[i]=l}else if(c===`bilinear`){let n=Math.floor(u),i=Math.ceil(u),s=u-n;for(let c=0;c<g;c++){let u=g>1?r*(f-1)+c*m:.5*(r+a)*(f-1);if(u<0||u>f-1){for(let n=0;n<p;n++){let r=n+c*S[2]+t*S[1]+e*S[0];_.values[r]=l}continue}let d=Math.floor(u),h=Math.ceil(u),v=u-d;for(let r=0;r<p;r++){let a=r+d*x[2]+n*x[1]+o*x[0],l=b[a];a=r+h*x[2]+n*x[1]+o*x[0];let u=b[a];a=r+d*x[2]+i*x[1]+o*x[0];let f=b[a];a=r+h*x[2]+i*x[1]+o*x[0];let p=b[a],m=l+(u-l)*v,g=f+(p-f)*v;a=r+c*S[2]+t*S[1]+e*S[0],_.values[a]=m+(g-m)*s}}}else for(let n=0;n<g;++n){let i=g>1?r*(f-1)+n*m:.5*(r+a)*(f-1);if(i<0||i>f-1){for(let r=0;r<p;r++){let i=r+n*S[2]+t*S[1]+e*S[0];_.values[i]=l}continue}let s=Math.round(i),c=Math.round(u);for(let r=0;r<p;r++){let i=r+s*x[2]+c*x[1]+o*x[0],a=r+n*S[2]+t*S[1]+e*S[0];_.values[a]=b[i]}}}}return n.makeTensorInfo(_.shape,_.dtype,_.values)}var Jx={kernelName:ot,backendName:`cpu`,kernelFunc:qx};function Yx(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r;G(i,`cumprod`);let c=_u([a],i.shape.length),l=i;c!=null&&(l=jy({inputs:{x:i},backend:n,attrs:{perm:c}}));let u=yu(1,i.shape.length)[0];if(u!==l.shape.length-1)throw Error(`backend.cumprod in CPU expects an inner-most axis=${l.shape.length-1} but got axis=${u}`);let d=xi(l.dtype,`int32`),f=ue(g(l.shape),d),p=n.data.get(l.dataId).values,m=l.shape[l.shape.length-1],h=s?(e,t)=>e+m-t-1:(e,t)=>e+t;for(let e=0;e<p.length;e+=m)for(let t=0;t<m;t++){let n=h(e,t);if(t===0)f[n]=o?1:p[n];else{let r=h(e,t-1);f[n]=o?p[r]*f[r]:p[n]*f[r]}}let _=n.makeTensorInfo(l.shape,d,f);if(c!=null){let e=vu(c),t=jy({inputs:{x:_},backend:n,attrs:{perm:e}});return n.disposeIntermediateTensorInfo(_),n.disposeIntermediateTensorInfo(l),t}return _}var Xx={kernelName:it,backendName:`cpu`,kernelFunc:Yx};function Zx(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r;G(i,`cumsum`);let c=_u([a],i.shape.length),l=i;c!=null&&(l=jy({inputs:{x:i},backend:n,attrs:{perm:c}}));let u=yu(1,i.shape.length)[0];if(u!==l.shape.length-1)throw Error(`backend.cumsum in CPU expects an inner-most axis=${l.shape.length-1} but got axis=${u}`);let d=xi(l.dtype,`int32`),f=M(g(l.shape),d),p=n.data.get(l.dataId).values,m=l.shape[l.shape.length-1],h=s?(e,t)=>e+m-t-1:(e,t)=>e+t;for(let e=0;e<p.length;e+=m)for(let t=0;t<m;t++){let n=h(e,t);if(t===0)f[n]=o?0:p[n];else{let r=h(e,t-1);f[n]=o?p[r]+f[r]:p[n]+f[r]}}let _=n.makeTensorInfo(l.shape,d,f);if(c!=null){let e=vu(c),t=jy({inputs:{x:_},backend:n,attrs:{perm:e}});return n.disposeIntermediateTensorInfo(_),n.disposeIntermediateTensorInfo(l),t}return _}var Qx={kernelName:at,backendName:`cpu`,kernelFunc:Zx};function $x(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,weights:a}=t,{size:o,binaryOutput:s}=r;if(i.shape.length===1){let e=n.data.get(i.dataId).values,t=n.data.get(a.dataId).values,r=Vv(e,t,a.dtype,a.shape,o);return n.makeTensorInfo([o],a.dtype,r)}if(i.shape.length===2){let e=Hv(n.bufferSync(i),n.bufferSync(a),o,s);return n.makeTensorInfo(e.shape,a.dtype,e.values)}throw Error(`Error in denseBincount: input must be at most rank 2, but got rank${i.shape.length}.`)}var eS={kernelName:st,backendName:`cpu`,kernelFunc:$x};function tS(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockSize:a,dataFormat:o}=r;f(o===`NHWC`,()=>`Only NHWC dataFormat supported on CPU for depthToSpace. Got ${o}`);let s=i.shape[0],c=i.shape[1],l=i.shape[2],u=i.shape[3],d=c*a,p=l*a,m=u/(a*a),h=n.data.get(i.dataId).values,g=new Float32Array(s*d*p*m),_=0;for(let e=0;e<s;++e)for(let t=0;t<d;++t){let n=Math.floor(t/a),r=t%a;for(let t=0;t<p;++t){let i=Math.floor(t/a),o=t%a,s=(r*a+o)*m;for(let t=0;t<m;++t){let r=t+s+u*(i+l*(n+c*e));g[_++]=h[r]}}}return n.makeTensorInfo([s,d,p,m],i.dtype,g)}var nS={kernelName:ct,backendName:`cpu`,kernelFunc:tS};function rS(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c,dimRoundingMode:l}=r;G([i,a],`depthwiseConv2DNative`);let u=j(i.shape),d=j(a.shape),p=c;p??=[1,1],f(Tc(o,p),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${o} and dilations '${p}'`);let m=pc(i.shape,a.shape,o,p,s,l,!0),{filterHeight:h,filterWidth:g,dilationHeight:_,dilationWidth:v,padInfo:y}=m,b=y.left,x=y.top,S=m.outChannels/m.inChannels,C=new si(m.outShape,i.dtype),w=n.data.get(i.dataId).values,T=n.data.get(a.dataId).values,E=C.values;for(let e=0;e<m.batchSize;++e){let t=e*u[0],n=e*C.strides[0];for(let e=0;e<m.outHeight;++e){let r=n+e*C.strides[1],i=e*m.strideHeight-x;for(let e=0;e<h;++e){let n=i+e*_;if(n<0||n>=m.inHeight)continue;let a=e*d[0],o=t+n*u[1];for(let e=0;e<m.outWidth;++e){let t=r+e*C.strides[2],n=e*m.strideWidth-b;for(let e=0;e<g;++e){let r=n+e*v;if(r<0||r>=m.inWidth)continue;let i=a+e*d[1],s=o+r*m.inChannels,c=t,l=i;for(let e=0;e<m.inChannels;++e){let t=w[s+e];for(let e=0;e<S;++e)E[c+e]+=t*T[l+e];c+=S,l+=S}}}}}}return n.makeTensorInfo(C.shape,C.dtype,C.values)}var iS={kernelName:lt,backendName:`cpu`,kernelFunc:rS};function aS(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,dilations:s,pad:c,dimRoundingMode:l,filterShape:u}=r;G([i,a],`depthwiseConv2dNativeBackpropFilter`);let d=pc(i.shape,u,o,s,c,l,!0),{strideHeight:f,strideWidth:p,filterHeight:m,filterWidth:h}=d,g=new si(d.filterShape,`float32`),_=d.padInfo.left,v=d.padInfo.top,y=d.outChannels/d.inChannels,b=n.data.get(i.dataId).values,x=new si(i.shape,i.dtype,b),S=n.data.get(a.dataId).values,C=new si(a.shape,a.dtype,S);for(let e=0;e<m;++e){let t=Math.max(0,Math.ceil((v-e)/f)),n=Math.min(d.outHeight,(d.inHeight+v-e)/f);for(let r=0;r<h;++r){let i=Math.max(0,Math.ceil((_-r)/p)),a=Math.min(d.outWidth,(d.inWidth+_-r)/p);for(let o=0;o<d.outChannels;++o){let s=Math.trunc(o/y),c=o%y,l=0;for(let c=0;c<d.batchSize;++c)for(let u=t;u<n;++u){let t=e+u*f-v;for(let e=i;e<a;++e){let n=r+e*p-_;l+=x.get(c,t,n,s)*C.get(c,u,e,o)}}g.set(l,e,r,s,c)}}}return n.makeTensorInfo(g.shape,g.dtype,g.values)}var oS={kernelName:ut,backendName:`cpu`,kernelFunc:aS};function sS(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{strides:o,dilations:s,pad:c,dimRoundingMode:l,inputShape:u}=r;G([i,a],`depthwiseConv2DNativeBackpropInput`);let d=j(i.shape),f=j(a.shape),p=pc(u,a.shape,o,s,c,l,!0),m=new si(p.inShape,`float32`),h=m.values,[g,_,v]=m.strides,y=n.data.get(i.dataId).values,[b,x,S]=d,C=n.data.get(a.dataId).values,[w,T,E]=f,{batchSize:D,filterHeight:O,filterWidth:ee,inChannels:k,inHeight:A,inWidth:te,outChannels:ne,outHeight:re,outWidth:ie,strideHeight:ae,strideWidth:oe}=p,se=O-1-p.padInfo.top,ce=ee-1-p.padInfo.left,le=ne/k;for(let e=0;e<D;++e)for(let t=0;t<k;++t)for(let n=0;n<A;++n){let r=n-se,i=Math.max(0,Math.ceil(r/ae)),a=Math.min(re,(O+r)/ae);for(let o=0;o<te;++o){let s=o-ce,c=Math.max(0,Math.ceil(s/oe)),l=Math.min(ie,(ee+s)/oe),u=0;for(let n=i;n<a;++n){let i=n*ae-r;for(let r=c;r<l;++r){let a=r*oe-s,o=b*e+x*n+S*r,c=w*(O-1-i)+T*(ee-1-a)+E*t;for(let e=0;e<le;++e){let n=y[o+(t*le+e)],r=C[c+e];u+=n*r}}}h[g*e+_*n+v*o+t]=u}}return n.makeTensorInfo(m.shape,m.dtype,m.values)}var cS={kernelName:dt,backendName:`cpu`,kernelFunc:sS};function lS(e){let{inputs:t,backend:n}=e,{x:r}=t,i=g(r.shape),a=n.data.get(r.dataId).values,o=L([i,i],r.dtype),s=o.values;for(let e=0;e<a.length;e++)s[e*i+e]=a[e];let c=[...r.shape,...r.shape];return n.makeTensorInfo(c,o.dtype,o.values)}var uS={kernelName:ft,backendName:`cpu`,kernelFunc:lS},dS={kernelName:pt,backendName:`cpu`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{let{x:r,filter:i}=e,{strides:a,pad:o,dilations:s}=n,c=t,l=c.data.get(r.dataId).values,u=r.shape.length,d=c.data.get(i.dataId).values,f=i.shape.length,{batchSize:p,inHeight:m,inWidth:h,inChannels:_,outHeight:v,outWidth:y,padInfo:b,strideHeight:x,strideWidth:S,filterHeight:C,filterWidth:w,dilationHeight:T,dilationWidth:D,outShape:O}=uc(r.shape,i.shape,a,o,`NHWC`,s),ee=g(O),k=O.length,A=E(r.dtype,ee);for(let e=0;e<p;++e)for(let t=0;t<v;++t){let n=t*x-b.top;for(let a=0;a<y;++a){let o=a*S-b.left;for(let s=0;s<_;++s){let c=-(2**53-1);for(let t=0;t<C;++t){let a=n+t*T;if(a>=0&&a<m)for(let n=0;n<w;++n){let p=o+n*D;if(p>=0&&p<h){let o=pe([e,a,p,s],u,j(r.shape)),m=pe([t,n,s],f,j(i.shape)),h=l[o]+d[m];h>c&&(c=h)}}}let p=pe([e,t,a,s],k,j(O));A[p]=c}}}return{dataId:c.write(Ur(A,r.dtype),O,r.dtype),shape:O,dtype:r.dtype}}},fS={kernelName:ht,backendName:`cpu`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{let{x:r,filter:i,dy:a}=e,{strides:o,pad:s,dilations:c}=n,l=t,u=le(r.shape,l.data.get(r.dataId).values),d=le(i.shape,l.data.get(i.dataId).values),{batchSize:p,inHeight:m,inWidth:h,inChannels:g,outHeight:_,outWidth:v,padInfo:y,strideHeight:b,strideWidth:x,filterHeight:S,filterWidth:C,dilationHeight:w,dilationWidth:T,outShape:E}=uc(r.shape,i.shape,o,s,`NHWC`,c);f(a.rank===E.length,()=>`Error in ${ht}, dy must have the same rank as output ${E.length}, but got ${a.rank}`);let D=le(E,l.data.get(a.dataId).values),O=de(i.shape,i.dtype);for(let e=0;e<p;++e)for(let t=0;t<_;++t){let n=t*b-y.top;for(let r=0;r<v;++r){let i=r*x-y.left;for(let a=0;a<g;++a){let o=-(2**53-1),s=0,c=0;for(let t=0;t<S;++t){let r=n+t*w;if(r>=0&&r<m)for(let n=0;n<C;++n){let l=i+n*T;if(l>=0&&l<h){let i=u[e][r][l][a]+d[t][n][a];i>o&&(o=i,s=t,c=n)}}}O[s][c][a]+=D[e][t][r][a]}}}return{dataId:l.write(Ur(O,r.dtype),i.shape,i.dtype),shape:i.shape,dtype:i.dtype}}},pS={kernelName:mt,backendName:`cpu`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{let{x:r,filter:i,dy:a}=e,{strides:o,pad:s,dilations:c}=n,l=t,u=le(r.shape,l.data.get(r.dataId).values),d=le(i.shape,l.data.get(i.dataId).values),{batchSize:p,inHeight:m,inWidth:h,inChannels:g,outHeight:_,outWidth:v,padInfo:y,strideHeight:b,strideWidth:x,filterHeight:S,filterWidth:C,dilationHeight:w,dilationWidth:T,outShape:E}=uc(r.shape,i.shape,o,s,`NHWC`,c);f(a.rank===E.length,()=>`Error in ${mt}, dy must have the same rank as output ${E.length}, but got ${a.rank}`);let D=le(E,l.data.get(a.dataId).values),O=de(r.shape,r.dtype);for(let e=0;e<p;++e)for(let t=0;t<_;++t){let n=t*b-y.top;for(let r=0;r<v;++r){let i=r*x-y.left;for(let a=0;a<g;++a){let o=-(2**53-1),s=n<0?0:n,c=i<0?0:i;for(let t=0;t<S;++t){let r=n+t*w;if(r>=0&&r<m)for(let n=0;n<C;++n){let l=i+n*T;if(l>=0&&l<h){let i=u[e][r][l][a]+d[t][n][a];i>o&&(o=i,s=r,c=l)}}}O[e][s][c][a]+=D[e][t][r][a]}}}return{dataId:l.write(Ur(O,r.dtype),r.shape,r.dtype),shape:r.shape,dtype:r.dtype}}};function mS(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;G(i,`sum`);let s;s=i.dtype===`bool`?Pv({inputs:{x:i},backend:n,attrs:{dtype:`int32`}}):kv({inputs:{x:i},backend:n});let c=s.shape.length,l=C(a,s.shape),u=_u(l,c),d=l,f=s;u!=null&&(f=jy({inputs:{x:s},backend:n,attrs:{perm:u}}),d=yu(d.length,c)),gu(`sum`,d,f.shape.length);let[p,m]=mu(f.shape,d),h=Ov(n,p,xi(f.dtype,`int32`)),_=g(m),v=n.data.get(h.dataId).values,y=n.data.get(f.dataId).values;for(let e=0;e<v.length;++e){let t=e*_,n=0;for(let e=0;e<_;++e)n+=y[t+e];v[e]=n}if(o){let e=hu(h.shape,l),t=h;h=q({inputs:{x:h},backend:n,attrs:{shape:e}}),n.disposeIntermediateTensorInfo(t)}return n.disposeIntermediateTensorInfo(s),u!=null&&n.disposeIntermediateTensorInfo(f),h}var hS={kernelName:`Sum`,backendName:`cpu`,kernelFunc:mS};function gS(e){let{inputs:t,backend:n,attrs:r}=e,{equation:i}=r,a=t,{allDims:o,summedDims:s,idDims:c}=Y_(i,a.length);Z_(o.length,c,a);let{path:l,steps:u}=Q_(s,c),d=u.length,f=null,p=o.length,m=[];for(let e=0;e<d;++e){for(let t of u[e]){let{permutationIndices:e,expandDims:r}=X_(p,c[t]),i;$_(e)?i=a[t]:(i=jy({inputs:{x:a[t]},backend:n,attrs:{perm:e}}),m.push(i));let o=i.shape.slice();for(let e=0;e<r.length;++e)o.splice(r[e],0,1);_(i.shape,o)||(i=q({inputs:{x:i},backend:n,attrs:{shape:o}}),m.push(i)),f===null?f=i:(f=Cy({inputs:{a:i,b:f},backend:n}),m.push(f))}e<d-1&&(l[e]>=0&&(f=mS({inputs:{x:f},backend:n,attrs:{axis:l[e]-(o.length-p),keepDims:!1}}),m.push(f)),p--)}for(let e of m)e!==f&&n.disposeIntermediateTensorInfo(e);return f}var _S={kernelName:_t,backendName:`cpu`,kernelFunc:gS};function vS(e){let{inputs:t,backend:n}=e,{dy:r,y:i}=t;G([r,i],`eluGrad`);let a=new Float32Array(g(i.shape)),o=n.data.get(i.dataId).values,s=n.data.get(r.dataId).values;for(let e=0;e<o.length;++e){let t=o[e];t>=1?a[e]=s[e]:a[e]=s[e]*(t+1)}return n.makeTensorInfo(i.shape,`float32`,a)}var yS={kernelName:vt,backendName:`cpu`,kernelFunc:vS},bS=j_,xS=M_,SS=N_,CS=P_,wS=F_,TS=I_,ES={kernelName:`Erf`,backendName:`cpu`,kernelFunc:K(`Erf`,e=>{let t=Math.sign(e),n=Math.abs(e),r=1/(1+bS*n);return t*(1-((((TS*r+wS)*r+CS)*r+SS)*r+xS)*r*Math.exp(-n*n))})};function DS(e){let{inputs:t,backend:n,attrs:r}=e,{input:i}=t,{dim:a}=r,o=i.shape.length,s=i.shape.slice(),c=a;return a<0&&(f(-(o+1)<=a,()=>`Axis must be in the interval [${-(o+1)}, ${o}]`),c=o+a+1),s.splice(c,0,1),q({inputs:{x:i},backend:n,attrs:{shape:s}})}var OS={kernelName:bt,backendName:`cpu`,kernelFunc:DS},kS=Iv(gt,Tv((e,t)=>e/t)),AS={kernelName:gt,backendName:`cpu`,kernelFunc:kS};function jS(e,t,n){let r=e.shape,i=r[0],a=r[1],o=n.data.get(e.dataId),s=o.complexTensorInfos.real,c=o.complexTensorInfos.imag,l=[i,a],u=g(l),d=T(`float32`,u),f=T(`float32`,u);for(let e=0;e<i;e++){let r=rb({inputs:{x:s},backend:n,attrs:{begin:[e,0],size:[1,a]}}),i=rb({inputs:{x:c},backend:n,attrs:{begin:[e,0],size:[1,a]}}),o=Ev({inputs:{real:r,imag:i},backend:n}),{real:l,imag:u}=MS(o,t,n),p=L_(l,u);for(let t=0;t<a;t++){let n=V_(p,t);d[e*a+t]=n.real,f[e*a+t]=n.imag}n.disposeIntermediateTensorInfo(r),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(o)}let p=n.makeTensorInfo(l,`float32`,d),m=n.makeTensorInfo(l,`float32`,f),h=Ev({inputs:{real:p,imag:m},backend:n});return n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),h}function MS(e,t,n){let r=g(e.shape),i=n.data.get(e.dataId),a=n.data.get(i.complexTensorInfos.real.dataId).values,o=n.data.get(i.complexTensorInfos.imag.dataId).values;if(NS(r)){let i=PS(a,o,r,t,n),s=[e.shape[0],e.shape[1]];if(t){let e=n.makeTensorInfo(s,`float32`,i.real),t=n.makeTensorInfo(s,`float32`,i.imag),a=n.makeTensorInfo([],`float32`,Vr(r,`float32`)),o=kv({inputs:{x:a},backend:n}),c=AS.kernelFunc({inputs:{a:e,b:a},backend:n}),l=AS.kernelFunc({inputs:{a:t,b:o},backend:n}),u=n.data.get(c.dataId).values,d=n.data.get(l.dataId).values;return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(a),n.disposeIntermediateTensorInfo(o),n.disposeIntermediateTensorInfo(c),n.disposeIntermediateTensorInfo(l),{real:u,imag:d}}return i}return R_(FS(L_(a,o),r,t))}function NS(e){return!(e&e-1)}function PS(e,t,n,r,i){if(n===1)return{real:e,imag:t};let a=L_(e,t),o=n/2,s=z_(a),c=s.real,l=s.imag,u=[c.length],d=i.makeTensorInfo(u,`float32`,c),f=i.makeTensorInfo(u,`float32`,l),p=Ev({inputs:{real:d,imag:f},backend:i}),m=B_(a),h=m.real,g=m.imag,_=[h.length],v=i.makeTensorInfo(_,`float32`,h),y=i.makeTensorInfo(_,`float32`,g),b=Ev({inputs:{real:v,imag:y},backend:i}),x=PS(c,l,o,r,i),S=x.real,C=x.imag,w=[S.length],T=i.makeTensorInfo(w,`float32`,S),E=i.makeTensorInfo(w,`float32`,C),D=Ev({inputs:{real:T,imag:E},backend:i}),O=PS(h,g,o,r,i),ee=O.real,k=O.imag,A=[ee.length],te=i.makeTensorInfo(A,`float32`,ee),ne=i.makeTensorInfo(A,`float32`,k),re=Ev({inputs:{real:te,imag:ne},backend:i}),ie=U_(n,r),ae=[ie.real.length],oe=i.makeTensorInfo(ae,`float32`,ie.real),se=i.makeTensorInfo(ae,`float32`,ie.imag),j=Ev({inputs:{real:oe,imag:se},backend:i}),ce=Cy({inputs:{a:j,b:re},backend:i}),le=zv({inputs:{a:D,b:ce},backend:i}),ue=yb({inputs:{a:D,b:ce},backend:i}),M=jv({inputs:{input:le},backend:i}),de=jv({inputs:{input:ue},backend:i}),fe=kx({inputs:{input:le},backend:i}),pe=kx({inputs:{input:ue},backend:i}),me=jx({inputs:[M,de],backend:i,attrs:{axis:0}}),he=jx({inputs:[fe,pe],backend:i,attrs:{axis:0}}),ge=i.data.get(me.dataId).values,_e=i.data.get(he.dataId).values;return i.disposeIntermediateTensorInfo(d),i.disposeIntermediateTensorInfo(f),i.disposeIntermediateTensorInfo(p),i.disposeIntermediateTensorInfo(v),i.disposeIntermediateTensorInfo(y),i.disposeIntermediateTensorInfo(b),i.disposeIntermediateTensorInfo(T),i.disposeIntermediateTensorInfo(E),i.disposeIntermediateTensorInfo(D),i.disposeIntermediateTensorInfo(te),i.disposeIntermediateTensorInfo(ne),i.disposeIntermediateTensorInfo(re),i.disposeIntermediateTensorInfo(oe),i.disposeIntermediateTensorInfo(se),i.disposeIntermediateTensorInfo(j),i.disposeIntermediateTensorInfo(ce),i.disposeIntermediateTensorInfo(le),i.disposeIntermediateTensorInfo(ue),i.disposeIntermediateTensorInfo(M),i.disposeIntermediateTensorInfo(fe),i.disposeIntermediateTensorInfo(de),i.disposeIntermediateTensorInfo(pe),i.disposeIntermediateTensorInfo(me),i.disposeIntermediateTensorInfo(he),{real:ge,imag:_e}}function FS(e,t,n){let r=new Float32Array(t*2);for(let i=0;i<t;i++){let a=0,o=0;for(let r=0;r<t;r++){let s=W_(i*r,t,n),c=V_(e,r);a+=c.real*s.real-c.imag*s.imag,o+=c.real*s.imag+c.imag*s.real}n&&(a/=t,o/=t),H_(r,a,o,i)}return r}function IS(e){let{inputs:t,backend:n}=e,{input:r}=t,i=g(r.shape),a=r.shape[r.shape.length-1],o=i/a,s=q({inputs:{x:r},backend:n,attrs:{shape:[o,a]}}),c=jS(s,!1,n),l=q({inputs:{x:c},backend:n,attrs:{shape:r.shape}});return n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(c),l}var LS={kernelName:`FFT`,backendName:`cpu`,kernelFunc:IS};function RS(e){let{backend:t,attrs:n}=e,{shape:r,value:i,dtype:a}=n,o=a||ae(i),s=E(o,g(r));return BS(s,i,o),t.makeTensorInfo(r,o,s)}var zS={kernelName:St,backendName:`cpu`,kernelFunc:RS};function BS(e,t,n){e.fill(t)}var VS={kernelName:Ct,backendName:`cpu`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{image:r}=e,i=n,a=T(r.dtype,g(r.shape)),[o,s,c,l]=r.shape,u=i.data.get(r.dataId).values;for(let e=0;e<o;e++){let t=e*c*s*l;for(let e=0;e<s;e++){let n=c*l*e;for(let e=0;e<c;e++){let r=e*l;for(let i=0;i<l;i++){let o=Math.round(c-e-1),s=t+n+r+i,d=u[s];if(o>=0&&o<c){let e=o*l;d=u[t+n+e+i]}a[s]=d}}}}return{dataId:i.write(a,r.shape,r.dtype),shape:r.shape,dtype:r.dtype}}},HS={kernelName:Tt,backendName:`cpu`,kernelFunc:Iv(Tt,Tv((e,t)=>Math.floor(e/t)),null,`int32`)};function US(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f,activation:p,leakyreluAlpha:m}=r,h=Nx({inputs:{x:i,filter:a},backend:n,attrs:{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f}});if(o){let e=h;if(u===`NCHW`&&o.shape.length===1&&o.shape[0]!==1){let e=q({inputs:{x:o},backend:n,attrs:{shape:[o.shape[0],1,1]}});h=zv({inputs:{a:h,b:e},backend:n}),n.disposeIntermediateTensorInfo(e)}else h=zv({inputs:{a:h,b:o},backend:n});n.disposeIntermediateTensorInfo(e)}if(p){let e=h;if(u===`NCHW`&&p===`prelu`&&s.shape.length===1&&s.shape[0]!==1){let e=q({inputs:{x:s},backend:n,attrs:{shape:[s.shape[0],1,1]}});h=Rb(n,h,p,e,m),n.disposeIntermediateTensorInfo(e)}else h=Rb(n,h,p,s,m);n.disposeIntermediateTensorInfo(e)}return h}var WS={kernelName:fr,backendName:`cpu`,kernelFunc:US};function GS(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f,activation:p,leakyreluAlpha:m}=r,h=rS({inputs:{x:i,filter:a},backend:n,attrs:{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f}});if(o){let e=h;h=zv({inputs:{a:h,b:o},backend:n}),n.disposeIntermediateTensorInfo(e)}if(p){let e=h;h=Rb(n,h,p,s,m),n.disposeIntermediateTensorInfo(e)}return h}var KS={kernelName:pr,backendName:`cpu`,kernelFunc:GS};function qS(e){let{inputs:t,backend:n}=e,{params:r,indices:i}=t,a=g(r.shape),o=i.shape,s=o[o.length-1],[c,l,u,d]=os(r,i);if(l===0)return n.makeTensorInfo(c,r.dtype,[]);let f=n.data.get(i.dataId).values,p=iy(f,n.bufferSync(r),r.dtype,l,s,u,d,r.shape,a);return n.makeTensorInfo(c,r.dtype,p.values)}var JS={kernelName:Ot,backendName:`cpu`,kernelFunc:qS};function YS(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,indices:a}=t,{axis:o,batchDims:s}=r;G([i,a],`gatherV2`);let c=C(o,i.shape)[0],l=n.data.get(a.dataId).values,u=i.shape[c];for(let e=0;e<l.length;++e){let t=l[e];f(t<=u-1&&t>=0,()=>`GatherV2: the index value ${t} is not in [0, ${u-1}]`)}let d=s;s??(d=0);let p=g(a.shape),m=_v(i,a,c,d),h=q({inputs:{x:i},backend:n,attrs:{shape:[m.batchSize,m.outerSize,m.dimSize,m.sliceSize]}}),_=q({inputs:{x:a},backend:n,attrs:{shape:[m.batchSize,p/m.batchSize]}}),v=[m.batchSize,m.outerSize,p/m.batchSize,m.sliceSize],y=n.bufferSync(_),b=ay(n.bufferSync(h),y,v);return n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(_),n.makeTensorInfo(m.outputShape,b.dtype,b.values)}var XS={kernelName:Dt,backendName:`cpu`,kernelFunc:YS};function ZS(e){let{inputs:t,backend:n}=e,{input:r}=t,i=g(r.shape),a=r.shape[r.shape.length-1],o=i/a,s=q({inputs:{x:r},backend:n,attrs:{shape:[o,a]}}),c=jS(s,!0,n),l=q({inputs:{x:c},backend:n,attrs:{shape:r.shape}});return n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(c),l}var QS={kernelName:Mt,backendName:`cpu`,kernelFunc:ZS},$S={kernelName:Pt,backendName:`cpu`,kernelFunc:K(Pt,e=>+!!Number.isFinite(e),`bool`)},eC={kernelName:Ft,backendName:`cpu`,kernelFunc:K(Ft,e=>+(Math.abs(e)===1/0),`bool`)},tC={kernelName:It,backendName:`cpu`,kernelFunc:K(It,e=>+!!Number.isNaN(e),`bool`)};function nC(e){let{backend:t,attrs:n}=e,{start:r,stop:i,num:a}=n,o=my(r,i,a);return t.makeTensorInfo([o.length],`float32`,o)}var rC={kernelName:Bt,backendName:`cpu`,kernelFunc:nC},iC={kernelName:Vt,backendName:`cpu`,kernelFunc:K(Vt,e=>Math.log1p(e))},aC={kernelName:Ht,backendName:`cpu`,kernelFunc:Iv(Ht,Tv((e,t)=>e&&t),null,`bool`)},oC={kernelName:Ut,backendName:`cpu`,kernelFunc:K(Ut,e=>+!e,`bool`)},sC={kernelName:Wt,backendName:`cpu`,kernelFunc:Iv(Wt,Tv((e,t)=>e||t),null,`bool`)};function cC(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{depthRadius:a,bias:o,alpha:s,beta:c}=r;G(i,`LRN`);let l=i.shape[3],u=l-1,d=n.data.get(i.dataId).values,f=g(i.shape),p=new Float32Array(f);function m(e){let t=e%l,n=e-t+Math.max(0,t-a),r=e-t+Math.min(t+a,u),i=0;for(;n<=r;n++){let e=d[n];i+=e*e}return i}for(let e=0;e<f;e++){let t=m(e),n=d[e]*(o+s*t)**+-c;p[e]=n}return n.makeTensorInfo(i.shape,i.dtype,p)}var lC={kernelName:`LRN`,backendName:`cpu`,kernelFunc:cC};function uC(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,y:a,dy:o}=t,{depthRadius:s,bias:c,alpha:l,beta:u}=r;G(o,`LRNGrad`);let d=g(o.shape),f=o.shape[3],p=n.data.get(o.dataId).values,m=n.data.get(i.dataId).values,h=n.data.get(a.dataId).values,_=new Float32Array(d),v=d;for(let e=0;e<v;e++){let t=e%f,n=e-t+Math.max(0,t-s),r=e-t+Math.min(f,t+s+1),i=0;for(let e=n;e<r;e++)i+=m[e]**2;i=l*i+c;for(let t=n;t<r;t++){let n=-2*l*u*m[t]*h[e]/i;e===t&&(n+=i**+-u),n*=p[e],_[t]+=n}}return n.makeTensorInfo(o.shape,i.dtype,_)}var dC={kernelName:Gt,backendName:`cpu`,kernelFunc:uC};function fC(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{reductionIndices:a,keepDims:o}=r,s=n,c=i.shape,l=c.length,u=C(a,c),d=u,f=_u(d,l),p=s.data.get(i.dataId).values;if(f!=null){let e=Array(l);for(let t=0;t<e.length;t++)e[t]=c[f[t]];p=Ay(p,c,i.dtype,f,e),d=yu(d.length,l),c=e}G(i,`max`),gu(`max`,d,l);let[m,h]=mu(c,d),_=g(h),v=_y(p,_,m,i.dtype),y=s.write(v,m,i.dtype),b=m;return o&&(b=hu(m,u)),{dataId:y,shape:b,dtype:i.dtype}}var pC={kernelName:`Max`,backendName:`cpu`,kernelFunc:fC};function mC(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t;G(i,`maxPool`);let{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=r;f(Tc(o,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${o} and dilations '1'`);let l=dc(i.shape,a,o,1,s,c),u;if(l.filterWidth===1&&l.filterHeight===1&&_(l.inShape,l.outShape))u=kv({inputs:{x:i},backend:n});else{let e=n.data.get(i.dataId).values,t=j(i.shape),r=sx(e,i.shape,i.dtype,t,l,`max`);u=n.makeTensorInfo(l.outShape,i.dtype,r.values)}return u}var hC={kernelName:qt,backendName:`cpu`,kernelFunc:mC};function gC(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c,dataFormat:l}=r;G(i,`maxPool3d`);let u=fc(i.shape,a,o,1,s,c,l),d=n.data.get(i.dataId).values,f=lx(d,i.shape,i.dtype,j(i.shape),u,`max`);return n.makeTensorInfo(f.shape,`float32`,f.values)}var _C={kernelName:Yt,backendName:`cpu`,kernelFunc:gC};function vC(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=r;G([i,a],`maxPool3DGrad`);let u=fc(a.shape,o,s,1,c,l),d=ux(n.bufferSync(a),u),f=u.strideDepth,p=u.strideHeight,m=u.strideWidth,h=u.dilationDepth,g=u.dilationHeight,_=u.dilationWidth,v=u.effectiveFilterDepth,y=u.effectiveFilterHeight,b=u.effectiveFilterWidth,x=v-1-u.padInfo.front,S=b-1-u.padInfo.left,C=y-1-u.padInfo.top,w=L(a.shape,`float32`),T=n.bufferSync(i);for(let e=0;e<u.batchSize;++e)for(let t=0;t<u.inChannels;++t)for(let n=0;n<u.inDepth;++n)for(let r=0;r<u.inHeight;++r)for(let i=0;i<u.inWidth;++i){let a=n-x,o=r-C,s=i-S,c=0;for(let n=0;n<v;n+=h){let r=(a+n)/f;if(!(r<0||r>=u.outDepth||Math.floor(r)!==r))for(let i=0;i<y;i+=g){let a=(o+i)/p;if(!(a<0||a>=u.outHeight||Math.floor(a)!==a))for(let o=0;o<b;o+=_){let l=(s+o)/m;if(l<0||l>=u.outWidth||Math.floor(l)!==l)continue;let f=+(v*y*b-1-d.get(e,r,a,l,t)===n*y*b+i*b+o);if(f===0)continue;let p=T.get(e,r,a,l,t);c+=p*f}}}w.set(c,e,n,r,i,t)}return n.makeTensorInfo(w.shape,w.dtype,w.values)}var yC={kernelName:Xt,backendName:`cpu`,kernelFunc:vC};function bC(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a,output:o}=t,s=a;G([a,o],`maxPoolGrad`);let{filterSize:c,strides:l,pad:u,dimRoundingMode:d}=r,f=dc(s.shape,c,l,1,u,d),p=n.data.get(s.dataId).values,m=L(f.outShape,s.dtype,cx(p,s.shape,s.dtype,f).values),h=f.strideHeight,g=f.strideWidth,_=f.dilationHeight,v=f.dilationWidth,y=f.effectiveFilterHeight,b=f.effectiveFilterWidth,x=b-1-f.padInfo.left,S=y-1-f.padInfo.top,C=L(s.shape,`float32`),w=n.data.get(i.dataId).values,T=L(i.shape,`float32`,w);for(let e=0;e<f.batchSize;++e)for(let t=0;t<f.inChannels;++t)for(let n=0;n<f.inHeight;++n)for(let r=0;r<f.inWidth;++r){let i=n-S,a=r-x,o=0;for(let n=0;n<y;n+=_){let r=(i+n)/h;if(!(r<0||r>=f.outHeight||Math.floor(r)!==r))for(let i=0;i<b;i+=v){let s=(a+i)/g;if(s<0||s>=f.outWidth||Math.floor(s)!==s)continue;let c=+(y*b-1-m.get(e,r,s,t)===n*b+i);if(c===0)continue;let l=T.get(e,r,s,t);o+=l*c}}C.set(o,e,n,r,t)}return n.makeTensorInfo(C.shape,C.dtype,C.values)}var xC={kernelName:Jt,backendName:`cpu`,kernelFunc:bC};function SC(e,t,n,r,i){let a=sx(e,t,n,j(t),i,`max`),o=cx(e,t,n,i,!0,r);return[a.values,o.values]}var CC={kernelName:Zt,backendName:`cpu`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{x:r}=e,{filterSize:i,strides:a,pad:o,includeBatchInIndex:s}=t,c=n;G(r,`MaxPoolWithArgmax`);let l=c.data.get(r.dataId).values,u=dc(r.shape,i,a,[1,1],o),[d,f]=SC(l,r.shape,r.dtype,s,u),p=c.write(d,u.outShape,r.dtype),m=c.write(f,u.outShape,r.dtype);return[{dataId:p,shape:u.outShape,dtype:r.dtype},{dataId:m,shape:u.outShape,dtype:`int32`}]}};function wC(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=C(a,i.shape),c=mu(i.shape,s)[1],l=g(c),u=[],d=n.makeTensorInfo([],`float32`,new Float32Array([l]));u.push(d);let f=Pv({inputs:{x:i},backend:n,attrs:{dtype:`float32`}});u.push(f);let p=kS({inputs:{a:f,b:d},backend:n});u.push(p);let m=mS({inputs:{x:p},backend:n,attrs:{axis:a,keepDims:o}});return u.forEach(e=>n.disposeIntermediateTensorInfo(e)),m}var TC={kernelName:Qt,backendName:`cpu`,kernelFunc:wC};function EC(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;G(i,`min`);let s=C(a,i.shape),c=s,l=_u(c,i.shape.length),u=i;l!=null&&(u=jy({inputs:{x:i},backend:n,attrs:{perm:l}}),c=yu(c.length,i.shape.length)),gu(`min`,c,u.shape.length);let[d,f]=mu(u.shape,c),p=g(f),m=M(g(d),u.dtype),h=n.data.get(u.dataId).values;for(let e=0;e<m.length;++e){let t=e*p,n=h[t];for(let e=0;e<p;++e){let r=h[t+e];(Number.isNaN(r)||r<n)&&(n=r)}m[e]=n}l!=null&&n.disposeIntermediateTensorInfo(u);let _=n.makeTensorInfo(d,u.dtype,m);if(o){let e=hu(d,s),t=q({inputs:{x:_},backend:n,attrs:{shape:e}});return n.disposeIntermediateTensorInfo(_),t}return _}var DC={kernelName:`Min`,backendName:`cpu`,kernelFunc:EC};function OC(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{paddings:a,mode:o}=r;G(i,`mirrorPad`);let s=a.map((e,t)=>e[0]+i.shape[t]+e[1]),c=a.map(e=>e[0]),l=a.map((e,t)=>e[0]+i.shape[t]),u=o===`reflect`?0:1,d=n.data.get(i.dataId).values,f=i.shape.length,p=j(i.shape),m=g(s),h=s.length,_=j(s),v=T(i.dtype,m);for(let e=0;e<m;e++){let t=me(e,h,_);for(let e=0;e<h;e++)t[e]<c[e]?t[e]=c[e]*2-t[e]-u:t[e]>=l[e]&&(t[e]=(l[e]-1)*2-t[e]+u);t=t.map((e,t)=>e-c[t]);let n=pe(t,f,p);v[e]=d[n]}return{dataId:n.write(v,s,i.dtype),shape:s,dtype:i.dtype}}var kC={kernelName:en,backendName:`cpu`,kernelFunc:OC},AC={kernelName:`Mod`,backendName:`cpu`,kernelFunc:Iv(`Mod`,Tv(((e,t)=>{let n=e%t;return e<0&&t<0||e>=0&&t>=0?n:(n+t)%t})))};function jC(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{dim:a}=r,o=i.shape.length,s=a;if(s===-1&&(s=o-1),s!==o-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${o} and dim was ${s}`);let c=C([s],i.shape),l=fC({inputs:{x:i},backend:n,attrs:{reductionIndices:c,keepDims:!1}}),u=hu(l.shape,c),d=q({inputs:{x:l},backend:n,attrs:{shape:u}}),f=yb({inputs:{a:i,b:d},backend:n}),p=Qv({inputs:{x:f},backend:n}),m=mS({inputs:{x:p},backend:n,attrs:{axis:c,keepDims:!1}}),h=q({inputs:{x:m},backend:n,attrs:{shape:u}}),g=kS({inputs:{a:p,b:h},backend:n});return n.disposeIntermediateTensorInfo(l),n.disposeIntermediateTensorInfo(d),n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(h),g}var MC={kernelName:Vn,backendName:`cpu`,kernelFunc:jC};function NC(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{numSamples:a,seed:o,normalized:s}=r;G(i,`multinomial`);let c=s?i:jC({inputs:{logits:i},backend:n,attrs:{dim:-1}}),l=c.shape[0],u=c.shape[1],d=n.data.get(c.dataId).values,f=[l,a],p=M(g(f),`int32`);for(let e=0;e<l;++e){let t=e*u,n=new Float32Array(u-1);n[0]=d[t];for(let e=1;e<n.length;++e)n[e]=n[e-1]+d[t+e];let r=$f.alea(o.toString()),i=e*a;for(let e=0;e<a;++e){let t=r();p[i+e]=n.length;for(let r=0;r<n.length;r++)if(t<n[r]){p[i+e]=r;break}}}return s||n.disposeIntermediateTensorInfo(c),n.makeTensorInfo(f,`int32`,p)}var PC={kernelName:tn,backendName:`cpu`,kernelFunc:NC},FC=Kh;function IC(e){let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c}=r;G(i,`NonMaxSuppression`);let l=n.data.get(i.dataId).values,u=n.data.get(a.dataId).values,{selectedIndices:d}=FC(l,u,o,s,c);return n.makeTensorInfo([d.length],`int32`,new Int32Array(d))}var LC={kernelName:an,backendName:`cpu`,kernelFunc:IC},RC=qh;function zC(e){let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c,padToMaxOutputSize:l}=r;G(i,`NonMaxSuppressionPadded`);let u=n.data.get(i.dataId).values,d=n.data.get(a.dataId).values,{selectedIndices:f,validOutputs:p}=RC(u,d,o,s,c,l);return[n.makeTensorInfo([f.length],`int32`,new Int32Array(f)),n.makeTensorInfo([],`int32`,new Int32Array([p]))]}var BC={kernelName:on,backendName:`cpu`,kernelFunc:zC},VC=Jh;function HC(e){let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c,softNmsSigma:l}=r;G(i,`NonMaxSuppressionWithScore`);let u=n.data.get(i.dataId).values,d=n.data.get(a.dataId).values,{selectedIndices:f,selectedScores:p}=VC(u,d,o,s,c,l);return[n.makeTensorInfo([f.length],`int32`,new Int32Array(f)),n.makeTensorInfo([p.length],`float32`,new Float32Array(p))]}var UC={kernelName:sn,backendName:`cpu`,kernelFunc:HC};function WC(e){let{inputs:t,backend:n,attrs:r}=e,{indices:i}=t,{dtype:a,depth:o,onValue:s,offValue:c}=r;G(i,`oneHot`);let l=g(i.shape),u=new Float32Array(l*o);u.fill(c);let d=n.data.get(i.dataId).values;for(let e=0;e<l;++e)d[e]>=0&&d[e]<o&&(u[e*o+d[e]]=s);return n.makeTensorInfo([...i.shape,o],a,u)}var GC={kernelName:ln,backendName:`cpu`,kernelFunc:WC};function KC(e){let{inputs:t,backend:n}=e,{x:r}=t;if(r.dtype===`string`)throw Error(`zerosLike is not supported for string tensors`);if(r.dtype===`complex64`){let e=jv({inputs:{input:r},backend:n}),t=KC({inputs:{x:e},backend:n}),i=kx({inputs:{input:r},backend:n}),a=KC({inputs:{x:i},backend:n}),o=Ev({inputs:{real:t,imag:a},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(a),o}return RS({backend:n,attrs:{shape:r.shape,value:0,dtype:r.dtype}})}var qC={kernelName:sr,backendName:`cpu`,kernelFunc:KC};function JC(e){let{inputs:t,backend:n}=e,{x:r}=t;if(r.dtype===`string`)throw Error(`onesLike is not supported for string tensors`);if(r.dtype===`complex64`){let e=jv({inputs:{input:r},backend:n}),t=JC({inputs:{x:e},backend:n}),i=kx({inputs:{input:r},backend:n}),a=KC({inputs:{x:i},backend:n}),o=Ev({inputs:{real:t,imag:a},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(a),o}return RS({backend:n,attrs:{shape:r.shape,value:1,dtype:r.dtype}})}var YC={kernelName:cn,backendName:`cpu`,kernelFunc:JC};function XC(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r;if(t.length===1)return DS({inputs:{input:t[0]},backend:n,attrs:{dim:i}});let a=t[0].shape,o=t[0].dtype;t.forEach(e=>{p(a,e.shape,`All tensors passed to stack must have matching shapes`),f(o===e.dtype,()=>`All tensors passed to stack must have matching dtypes`)});let s=[],c=jx({inputs:t.map(e=>{let t=DS({inputs:{input:e},backend:n,attrs:{dim:i}});return s.push(t),t}),backend:n,attrs:{axis:i}});return s.forEach(e=>n.disposeIntermediateTensorInfo(e)),c}var ZC={kernelName:un,backendName:`cpu`,kernelFunc:XC};function QC(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{paddings:a,constantValue:o}=r;G(i,`pad`);let s=a.map((e,t)=>e[0]+i.shape[t]+e[1]),c=a.map(e=>e[0]),l=n.data.get(i.dataId).values,u=g(i.shape),d=i.shape.length,f=j(i.shape),p=g(s),m=s.length,h=j(s),_=T(i.dtype,p);o!==0&&_.fill(o);for(let e=0;e<u;e++){let t=pe(me(e,d,f).map((e,t)=>e+c[t]),m,h);_[t]=l[e]}return{dataId:n.write(_,s,i.dtype),shape:s,dtype:i.dtype}}var $C={kernelName:dn,backendName:`cpu`,kernelFunc:QC},ew={kernelName:`Pow`,backendName:`cpu`,kernelFunc:Iv(`Pow`,Tv((e,t)=>e**+t))};function tw(e){let{inputs:t,backend:n,attrs:r}=e,{paramsNestedSplits:i,paramsDenseValues:a,indices:o}=t,{outputRaggedRank:s}=r,c=i.map(e=>n.data.get(e.dataId).values),l=i.map(e=>e.shape),u=n.data.get(a.dataId).values,d=n.data.get(o.dataId).values,[f,p,m]=Uy(c,l,u,a.shape,a.dtype,d,o.shape,s),h=f.map(e=>n.makeTensorInfo([e.length],`int32`,e)),g=n.makeTensorInfo(m,a.dtype,p);return h.concat([g])}var nw={kernelName:mn,backendName:`cpu`,kernelFunc:tw};function rw(e){let{inputs:t,backend:n,attrs:r}=e,{shape:i,values:a,defaultValue:o,rowPartitionTensors:s}=t,{rowPartitionTypes:c}=r,l=n.data.get(i.dataId).values,u=n.data.get(a.dataId).values,d=n.data.get(o.dataId).values,f=s.map(e=>n.data.get(e.dataId).values),p=s.map(e=>e.shape),[m,h]=Jy(l,i.shape,u,a.shape,a.dtype,d,o.shape,f,p,c);return n.makeTensorInfo(m,a.dtype,h)}var iw={kernelName:hn,backendName:`cpu`,kernelFunc:rw};function aw(e){let{backend:t,attrs:n}=e,{start:r,stop:i,dtype:a,step:o}=n,s=Yy(r,i,o,a);return t.makeTensorInfo([s.length],a,s)}var ow={kernelName:gn,backendName:`cpu`,kernelFunc:aw},sw={kernelName:vn,backendName:`cpu`,kernelFunc:K(vn,e=>1/e)};function cw(e){let{inputs:t,backend:n,attrs:r}=e,{images:i}=t,{alignCorners:a,halfPixelCenters:o,size:s}=r;G(i,`resizeBilinear`);let c=j(i.shape),[l,u]=s,[d,f,p,m]=i.shape,h=n.data.get(i.dataId).values,_=new Float32Array(g([d,l,u,m])),v=[a&&l>1?f-1:f,a&&u>1?p-1:p],y=[a&&l>1?l-1:l,a&&u>1?u-1:u],b=0,x=v[0]/y[0],S=v[1]/y[1];for(let e=0;e<d;e++)for(let t=0;t<l;t++){let n;n=o?x*(t+.5)-.5:x*t;let r=Math.max(0,Math.floor(n)),i=n-r,a=Math.min(f-1,Math.ceil(n)),s=e*c[0]+r*c[1],l=e*c[0]+a*c[1];for(let e=0;e<u;e++){let t;t=o?S*(e+.5)-.5:S*e;let n=Math.max(0,Math.floor(t)),r=t-n,a=Math.min(p-1,Math.ceil(t)),u=s+n*c[2],d=l+n*c[2],f=s+a*c[2],g=l+a*c[2];for(let e=0;e<m;e++){let t=h[u+e],n=h[d+e],a=h[f+e],o=h[g+e],s=t+(a-t)*r,c=s+(n+(o-n)*r-s)*i;_[b++]=c}}}return n.makeTensorInfo([d,l,u,m],`float32`,_)}var lw={kernelName:Cn,backendName:`cpu`,kernelFunc:cw};function uw(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r;G([a,i],`resizeBilinearGrad`);let s=j(i.shape),[c,l,u,d]=i.shape,[,f,p]=a.shape,m=new Float32Array(c*l*u*d),h=[o&&f>1?l-1:l,o&&p>1?u-1:u],g=[o&&f>1?f-1:f,o&&p>1?p-1:p],_=h[0]/g[0],v=h[1]/g[1],y=n.data.get(a.dataId).values,b=0;for(let e=0;e<c;e++){let t=e*s[0];for(let e=0;e<f;e++){let n=e*_,r=Math.floor(n),i=Math.min(Math.ceil(n),l-1),a=t+r*s[1],o=t+i*s[1],c=n-r,f=1-c;for(let e=0;e<p;e++){let t=e*v,n=Math.floor(t),r=Math.min(Math.ceil(t),u-1),i=t-n,l=1-i,p=a+n*s[2],h=a+r*s[2],g=o+n*s[2],_=o+r*s[2],x=f*l,S=f*i,C=c*l,w=c*i;for(let e=0;e<d;e++){let t=y[b++];m[p+e]+=t*x,m[h+e]+=t*S,m[g+e]+=t*C,m[_+e]+=t*w}}}}return n.makeTensorInfo([c,u,l,d],`float32`,m)}var dw={kernelName:wn,backendName:`cpu`,kernelFunc:uw};function fw(e){let{inputs:t,backend:n,attrs:r}=e,{images:i}=t,{alignCorners:a,halfPixelCenters:o,size:s}=r;G(i,`resizeNearestNeighbor`);let c=j(i.shape),[l,u]=s,[d,f,p,m]=i.shape,h=n.data.get(i.dataId).values,g=new Float32Array(d*l*u*m),_=[a&&l>1?f-1:f,a&&u>1?p-1:p],v=[a&&l>1?l-1:l,a&&u>1?u-1:u],y=_[0]/v[0],b=_[1]/v[1],x=0;for(let e=0;e<d;e++){let t=e*c[0];for(let e=0;e<l;e++){let n=o?y*(e+.5):y*e,r=Math.min(f-1,a?Math.round(n):Math.floor(n));o&&(r=Math.max(0,r));let i=t+r*c[1];for(let e=0;e<u;e++){let t=o?b*(e+.5):b*e,n=Math.min(p-1,a?Math.round(t):Math.floor(t));o&&(n=Math.max(0,n));let r=i+n*c[2];for(let e=0;e<m;e++){let t=h[r+e];g[x++]=t}}}}return n.makeTensorInfo([d,l,u,m],i.dtype,g)}var pw={kernelName:xn,backendName:`cpu`,kernelFunc:fw};function mw(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r;G([a,i],`resizeNearestNeighborGrad`);let s=j(i.shape),c=j(a.shape),[l,u,d,f]=i.shape,[,p,m]=a.shape,h=new Float32Array(l*u*d*f),g=n.data.get(a.dataId).values,_=[o&&p>1?u-1:u,o&&m>1?d-1:d],v=[o&&p>1?p-1:p,o&&m>1?m-1:m],y=_[0]/v[0],b=_[1]/v[1],x=1/y,S=1/b,C=Math.ceil(x)*2+2,w=Math.ceil(S)*2+2;for(let e=0;e<l;e++){let t=e*s[0];for(let e=0;e<u;e++){let n=t+e*s[1],r=Math.floor(e*x),i=Math.floor(r-C/2);for(let r=0;r<d;r++){let a=n+r*s[2],l=Math.floor(r*S),_=Math.floor(l-w/2);for(let n=0;n<f;n++){let s=0;for(let a=0;a<C;a++){let l=a+i;if(l<0||l>=p)continue;let f=t+l*c[1],h=l*y,v=Math.min(u-1,o?Math.round(h):Math.floor(h));if(e===v)for(let e=0;e<w;e++){let t=e+_;if(t<0||t>=m)continue;let i=f+t*c[2],a=t*b,l=Math.min(d-1,o?Math.round(a):Math.floor(a));r===l&&(s+=g[i+n])}}h[a+n]=s}}}}return n.makeTensorInfo(i.shape,i.dtype,h)}var hw={kernelName:Sn,backendName:`cpu`,kernelFunc:mw};function gw(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dims:a}=r;G(i,`reverse`);let o=i.shape.length,s=C(a,i.shape);if(o===0)return kv({inputs:{x:i},backend:n});let c=new si(i.shape,i.dtype),l=n.bufferSync(i);for(let e=0;e<c.size;e++){let t=c.indexToLoc(e),n=t.slice();s.forEach(e=>n[e]=i.shape[e]-1-n[e]),c.set(l.get(...n),...t)}return n.makeTensorInfo(c.shape,c.dtype,c.values)}var _w={kernelName:En,backendName:`cpu`,kernelFunc:gw},vw={kernelName:ur,backendName:`cpu`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{image:r}=e,{radians:i,fillValue:a,center:o}=t,s=n,c=T(r.dtype,g(r.shape)),[l,u,d,f]=r.shape,[p,m]=C_(o,u,d),h=Math.sin(i),_=Math.cos(i),v=s.data.get(r.dataId).values;for(let e=0;e<l;e++){let t=e*d*u*f;for(let e=0;e<u;e++){let n=d*f*e;for(let r=0;r<d;r++){let i=r*f;for(let o=0;o<f;o++){let s=[l,e,r,o],g=s[2],y=s[1],b=(g-p)*_-(y-m)*h,x=(g-p)*h+(y-m)*_;b=Math.round(b+p),x=Math.round(x+m);let S=a;if(typeof a!=`number`&&(S=o===3?255:a[o]),b>=0&&b<d&&x>=0&&x<u){let e=d*f*x,n=b*f;S=v[t+e+n+o]}let C=t+n+i+o;c[C]=S}}}}return{dataId:s.write(c,r.shape,r.dtype),shape:r.shape,dtype:r.dtype}}},yw={kernelName:Dn,backendName:`cpu`,kernelFunc:K(Dn,e=>{let t=Math.floor(e);return e-t<.5?Math.floor(e):e-t>.5?Math.ceil(e):t%2==0?t:t+1})};function bw(e){let{inputs:t,backend:n,attrs:r}=e,{indices:i,updates:a}=t,{shape:o}=r,{sliceRank:s,numUpdates:c,sliceSize:l,strides:u,outputSize:d}=ls(a,i,o),f=Qy(n.bufferSync(i),n.bufferSync(a),o,d,l,c,s,u,0,!0);return n.makeTensorInfo(o,f.dtype,f.values)}var xw={kernelName:kn,backendName:`cpu`,kernelFunc:bw};function Sw(e,t){let n=0,r=e.length,i=0;for(;n<r;)i=Math.floor((n+r)/2),e[i]<t?n=i+1:r=i;return r}function Cw(e,t){let n=0,r=e.length,i=0;for(;n<r;)i=Math.floor((n+r)/2),e[i]<=t?n=i+1:r=i;return r}function ww(e,t,n,r,i,a){let o=E(`int32`,n*i);for(let s=0;s<n;++s){let n=e.slice(s*r,(s+1)*r),c=s*i;for(let e=0;e<i;++e)o[c+e]=a===`left`?Sw(n,t[e+c]):Cw(n,t[e+c])}return o}function Tw(e){let{inputs:t,backend:n,attrs:r}=e,{sortedSequence:i,values:a}=t,{side:o}=r,s=n.data.get(i.dataId).values,c=n.data.get(a.dataId).values,l=ww(s,c,i.shape[0],i.shape[1],a.shape[1],o);return n.makeTensorInfo(a.shape,`int32`,l)}var Ew={kernelName:An,backendName:`cpu`,kernelFunc:Tw};function Dw(e){let{inputs:t,backend:n}=e,{condition:r,t:i,e:a}=t;G([r,i,a],`select`);let o=r.shape.length,s=n.data.get(r.dataId).values,c=n.data.get(i.dataId).values,l=n.data.get(a.dataId).values,u=xi(i.dtype,a.dtype),d=M(g(i.shape),u),f=0,p=o===0||o>1||i.shape.length===1?1:g(i.shape.slice(1));for(let e=0;e<s.length;e++)for(let t=0;t<p;t++)s[e]===1?d[f++]=c[e]:d[f++]=l[e];return n.makeTensorInfo(i.shape,u,d)}var Ow={kernelName:jn,backendName:`cpu`,kernelFunc:Dw},kw=k_,Aw=A_,jw={kernelName:Mn,backendName:`cpu`,kernelFunc:K(Mn,e=>e>=0?Aw*e:kw*(Math.exp(e)-1))},Mw={kernelName:Fn,backendName:`cpu`,kernelFunc:K(Fn,e=>e<0?-1:+(e>0))},Nw={kernelName:`Sin`,backendName:`cpu`,kernelFunc:K(`Sin`,e=>Math.sin(e))},Pw={kernelName:Pn,backendName:`cpu`,kernelFunc:K(Pn,e=>Math.sinh(e))},Fw=Math.log(1.1920928955078125e-7)+2,Iw={kernelName:Ln,backendName:`cpu`,kernelFunc:K(Ln,e=>{let t=e>-Fw,n=e<Fw,r=Math.exp(e),i;return i=n?r:t?e:Math.log(1+r),i})};function Lw(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,paddings:o}=r;G([i],`spaceToBatchND`);let s=g(a),c=[[0,0]];c.push(...o);for(let e=1+a.length;e<i.shape.length;++e)c.push([0,0]);let l=$C.kernelFunc({inputs:{x:i},backend:n,attrs:{paddings:c,constantValue:0}}),u=w_(l.shape,a,s,!1),d=T_(u.length,a.length,!1),f=E_(l.shape,a,s,!1),p=q({inputs:{x:l},backend:n,attrs:{shape:u}}),m=jy({inputs:{x:p},backend:n,attrs:{perm:d}}),h=q({inputs:{x:m},backend:n,attrs:{shape:f}});return n.disposeIntermediateTensorInfo(l),n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),h}var Rw={kernelName:zn,backendName:`cpu`,kernelFunc:Lw};function zw(e){let{inputs:t,backend:n}=e,{indices:r,values:i,denseShape:a,defaultValue:o}=t;if(a.shape.length!==1)throw Error(`Dense shape must be a vector, saw:
        ${a.shape}`);if(r.shape.length!==2)throw Error(`Indices must be a matrix, saw:
        ${r.shape}`);if(i.shape.length!==1)throw Error(`Values must be a vector, saw:
        ${i.shape}`);if(o.shape.length!==0)throw Error(`Default value must be a scalar, saw:
        ${o.shape}`);let s=n.data.get(r.dataId).values,c=n.data.get(i.dataId).values,l=n.data.get(a.dataId).values,u=n.data.get(o.dataId).values[0],[d,f,p,m,h]=ab(s,r.shape,r.dtype,c,i.dtype,l,u);return[n.makeTensorInfo(f,r.dtype,d),n.makeTensorInfo([f[0]],i.dtype,p),n.makeTensorInfo([m.length],`bool`,new Uint8Array(m.map(e=>Number(e)))),n.makeTensorInfo([h.length],r.dtype,new Int32Array(h))]}var Bw={kernelName:Hn,backendName:`cpu`,kernelFunc:zw};function Vw(e){let{inputs:t,backend:n}=e,{inputIndices:r,inputShape:i,newShape:a}=t;if(r.shape.length!==2)throw Error(`Input indices should be a matrix but received shape
        ${r.shape}`);if(i.shape.length!==1)throw Error(`Input shape should be a vector but received shape
        ${i.shape}`);if(a.shape.length!==1)throw Error(`Target shape should be a vector but received shape ${a.shape}`);let o=Array.from(n.data.get(i.dataId).values),s=n.data.get(r.dataId).values,c=Array.from(n.data.get(a.dataId).values),[l,u,d]=ob(s,r.shape,r.dtype,o,c);return[n.makeTensorInfo(u,r.dtype,l),n.makeTensorInfo([d.length],a.dtype,new Int32Array(d))]}var Hw={kernelName:Un,backendName:`cpu`,kernelFunc:Vw};function Uw(e){let{inputs:t,backend:n}=e,{data:r,indices:i,segmentIds:a}=t;if(r.shape.length<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.shape.length!==1)throw Error(`Indices should be a vector but received shape
          ${i.shape}`);if(a.shape.length!==1)throw Error(`Segment ids should be a vector but received shape
          ${a.shape}`);if(i.shape[0]!==a.shape[0])throw Error(`segmentIds and indices should have same size.`);let o=n.data.get(r.dataId).values,s=n.data.get(i.dataId).values,c=n.data.get(a.dataId).values,[l,u]=sb(o,r.shape,r.dtype,s,c,!0);return n.makeTensorInfo(u,r.dtype,l)}var Ww={kernelName:Wn,backendName:`cpu`,kernelFunc:Uw};function Gw(e){let{inputs:t,backend:n}=e,{data:r,indices:i,segmentIds:a}=t;if(r.shape.length<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.shape.length!==1)throw Error(`Indices should be a vector but received shape
         ${i.shape}`);if(a.shape.length!==1)throw Error(`Segment ids should be a vector but received shape
         ${a.shape}`);if(i.shape[0]!==a.shape[0])throw Error(`segmentIds and indices should have same size.`);let o=n.data.get(r.dataId).values,s=n.data.get(i.dataId).values,c=n.data.get(a.dataId).values,[l,u]=sb(o,r.shape,r.dtype,s,c);return n.makeTensorInfo(u,r.dtype,l)}var Kw={kernelName:Gn,backendName:`cpu`,kernelFunc:Gw};function qw(e){let{inputs:t,backend:n,attrs:r}=e,{sparseIndices:i,sparseValues:a,defaultValue:o}=t,{outputShape:s}=r,{sliceRank:c,numUpdates:l,sliceSize:u,strides:d,outputSize:f}=ls(a,i,s),p=n.bufferSync(i),m;switch(a.dtype){case`bool`:m=Qy(p,n.bufferSync(a),s,f,u,l,c,d,!!n.data.get(o.dataId).values[0],!1);break;case`float32`:{let e=n.bufferSync(a),t=n.data.get(o.dataId).values[0];m=Qy(p,e,s,f,u,l,c,d,t,!1);break}case`int32`:{let e=n.bufferSync(a),t=n.data.get(o.dataId).values[0];m=Qy(p,e,s,f,u,l,c,d,t,!1);break}case`string`:m=Qy(p,n.bufferSync(a),s,f,u,l,c,d,Kr(n.data.get(o.dataId).values[0]),!1);break;default:throw Error(`Unsupported type ${a.dtype}`)}return n.makeTensorInfo(s,m.dtype,m.values)}var Jw={kernelName:Kn,backendName:`cpu`,kernelFunc:qw};function Yw(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{numOrSizeSplits:a,axis:o}=r,s=C(o,i.shape)[0],c=tv(i,a,s),l=Array(i.shape.length).fill(0),u=i.shape.slice();return c.map(e=>{let t=[...u];t[s]=e;let r=rb({inputs:{x:i},backend:n,attrs:{begin:l,size:t}});return l[s]+=e,r})}var Xw={kernelName:Bn,backendName:`cpu`,kernelFunc:Yw},Zw={kernelName:Jn,backendName:`cpu`,kernelFunc:({inputs:e,backend:t})=>{let{x:n}=e,r=t;G(n,`square`);let i=r.data.get(n.dataId).values,a=new Float32Array(i.length);for(let e=0;e<i.length;++e){let t=i[e];a[e]=t*t}return{dataId:r.write(a,n.shape,n.dtype),shape:n.shape,dtype:n.dtype}}},Qw={kernelName:cr,backendName:`cpu`,kernelFunc:K(cr,(e,t)=>isNaN(e)?NaN:e>0?1:t.alpha)};function $w(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{begin:a,end:o,strides:s,beginMask:c,endMask:l,ellipsisMask:u,newAxisMask:d,shrinkAxisMask:p}=r;G(i,`stridedSlice`);let{finalShapeSparse:m,finalShape:h,isIdentity:g,sliceDim0:_,isSimpleSlice:v,begin:y,end:b,strides:x}=Os(i.shape,a,o,s,c,l,u,d,p),S;if(g)S=q({inputs:{x:i},backend:n,attrs:{shape:h}});else if(_||v){f(i.shape.length>=1,()=>`Input must have rank at least 1, got: ${i.shape.length}`);let e=hs(y,b,x),t=rb({inputs:{x:i},backend:n,attrs:{begin:y,size:e}});S=q({inputs:{x:t},backend:n,attrs:{shape:h}}),n.disposeIntermediateTensorInfo(t)}else{let e=fb(m,n.bufferSync(i),x,y);S=n.makeTensorInfo(h,e.dtype,e.values)}return S}var eT={kernelName:Yn,backendName:`cpu`,kernelFunc:$w};function tT(e){let{inputs:t,backend:n,attrs:r}=e,{separator:i,nGramWidths:a,leftPad:o,rightPad:s,padWidth:c,preserveShortSequences:l}=r,{data:u,dataSplits:d}=t,f=n.data.get(u.dataId).values,p=n.data.get(d.dataId).values,[m,h]=mb(f,p,i,a,o,s,c,l);return[n.makeTensorInfo([m.length],`string`,m),n.makeTensorInfo(d.shape,`int32`,h)]}var nT={kernelName:Xn,backendName:`cpu`,kernelFunc:tT};function rT(e){let{inputs:t,backend:n,attrs:r}=e,{skipEmpty:i}=r,{input:a,delimiter:o}=t;if(a.dtype!==`string`)throw Error(`Input must be of datatype string`);if(a.shape.length!==1)throw Error(`Input must be a vector, got shape: ${a.shape}`);if(o.shape.length!==0)throw Error(`Delimiter must be a scalar, got shape: ${o.shape}`);let s=n.data.get(a.dataId).values,c=n.data.get(o.dataId).values[0],[l,u,d]=gb(s,c,i),f=u.length;return[n.makeTensorInfo([f,2],`int32`,l),n.makeTensorInfo([f],`string`,u),n.makeTensorInfo([2],`int32`,new Int32Array(d))]}var iT={kernelName:Zn,backendName:`cpu`,kernelFunc:rT};function aT(e){let{inputs:t,backend:n,attrs:r}=e,{numBuckets:i}=r,{input:a}=t;if(a.dtype!==`string`)throw Error(`Input must be of datatype string`);if(i<=0)throw Error(`Number of buckets must be at least 1`);let o=n.data.get(a.dataId).values,s=_b(o,i);return n.makeTensorInfo(a.shape,`int32`,s)}var oT={kernelName:Qn,backendName:`cpu`,kernelFunc:aT},sT={kernelName:`Tan`,backendName:`cpu`,kernelFunc:K(`Tan`,e=>Math.tan(e))},cT={kernelName:$n,backendName:`cpu`,kernelFunc:K($n,e=>Math.tanh(e))};function lT(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{reps:a}=r;G(i,`tile`);let o=xb(n.bufferSync(i),a);return n.makeTensorInfo(o.shape,o.dtype,o.values)}var uT={kernelName:er,backendName:`cpu`,kernelFunc:lT};function dT(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{k:a,sorted:o}=r;G(i,`topk`);let s=n.data.get(i.dataId).values,[c,l]=wb(s,i.shape,i.dtype,a,o);return[n.makeTensorInfo(c.shape,c.dtype,c.values),n.makeTensorInfo(l.shape,l.dtype,l.values)]}var fT={kernelName:tr,backendName:`cpu`,kernelFunc:dT};function pT(e){let{inputs:t,attrs:n,backend:r}=e,{image:i,transforms:a}=t,{interpolation:o,fillMode:s,fillValue:c,outputShape:l}=n,[u,d,f,p]=i.shape,[m,h]=l??[d,f],_=[u,m,h,p],v=j(i.shape),y=v[0],b=v[1],x=v[2],S=j(_),C=S[0],w=S[1],E=S[2],D=T(i.dtype,g(_));D.fill(c);let O=r.data.get(i.dataId).values,ee=r.data.get(a.dataId).values;for(let e=0;e<u;++e){let t=a.shape[0]===1?ee:ee.subarray(e*8,e*8+8);for(let n=0;n<m;++n)for(let r=0;r<h;++r)for(let i=0;i<p;++i){let a,l=t[6]*r+t[7]*n+1;if(l===0)continue;let u=(t[0]*r+t[1]*n+t[2])/l,p=(t[3]*r+t[4]*n+t[5])/l,m=hT(u,f,s),h=hT(p,d,s);switch(o){case`nearest`:a=xT(O,d,f,y,b,x,e,h,m,i,c);break;case`bilinear`:a=ST(O,d,f,y,b,x,e,h,m,i,c);break;default:throw Error(`Error in Transform: Expect 'nearest' or 'bilinear', but got ${o}`)}let g=e*C+n*w+r*E+i;D[g]=a}return r.makeTensorInfo(_,i.dtype,D)}return{dataId:r.write(D,_,i.dtype),shape:i.shape,dtype:i.dtype}}var mT={kernelName:nr,backendName:`cpu`,kernelFunc:pT};function hT(e,t,n){switch(n){case`reflect`:return gT(e,t);case`wrap`:return _T(e,t);case`nearest`:return yT(e,t);default:return vT(e,t)}}function gT(e,t){let n=e;if(n<0){if(t<=1)n=0;else{let e=2*t;n<e&&(n=e*Math.trunc(-n/e)+n),n=n<-t?n+e:-n-1}}else if(n>t-1){if(t<=1)n=0;else{let e=2*t;n-=e*Math.trunc(n/e),n>=t&&(n=e-n-1)}}return c(0,n,t-1)}function _T(e,t){let n=e;if(n<0){if(t<=1)n=0;else{let e=t-1;n+=t*(Math.trunc(-n/e)+1)}}else if(n>t-1){if(t<=1)n=0;else{let e=t-1;n-=t*Math.trunc(n/e)}}return c(0,n,t-1)}function vT(e,t){return e}function yT(e,t){return c(0,e,t-1)}function bT(e,t,n,r,i,a,o,s,c,l,u){let d=o*r+s*i+c*a+l;return 0<=s&&s<t&&0<=c&&c<n?e[d]:u}function xT(e,t,n,r,i,a,o,s,c,l,u){return bT(e,t,n,r,i,a,o,Math.round(s),Math.round(c),l,u)}function ST(e,t,n,r,i,a,o,s,c,l,u){let d=Math.floor(s),f=Math.floor(c),p=d+1,m=f+1,h=(m-c)*bT(e,t,n,r,i,a,o,d,f,l,u)+(c-f)*bT(e,t,n,r,i,a,o,d,m,l,u),g=(m-c)*bT(e,t,n,r,i,a,o,p,f,l,u)+(c-f)*bT(e,t,n,r,i,a,o,p,m,l,u);return(p-s)*h+(s-d)*g}function CT(e){let{inputs:t,attrs:n,backend:r}=e,{axis:i}=n,{x:a}=t;G(a,`unique`);let o=r.data.get(a.dataId).values,{outputValues:s,outputShape:c,indices:l}=Tb(o,i,a.shape,a.dtype);return[r.makeTensorInfo(c,a.dtype,s),r.makeTensorInfo([l.length],`int32`,l)]}var wT={kernelName:ir,backendName:`cpu`,kernelFunc:CT};function TT(e){let{inputs:t,backend:n,attrs:r}=e,{value:i}=t,{axis:a}=r;a<0&&(a+=i.shape.length);let o=i.shape.length,s=i.shape[a],c=Array(o-1),l=0;for(let e=0;e<o;e++)e!==a&&(c[l++]=i.shape[e]);let u=Array(o).fill(0),d=i.shape.slice();d[a]=1;let f=Array(s);for(let e=0;e<f.length;e++){u[a]=e;let t=rb({inputs:{x:i},backend:n,attrs:{begin:u,size:d}});f[e]=q({inputs:{x:t},backend:n,attrs:{shape:c}}),n.disposeIntermediateTensorInfo(t)}return f}var ET={kernelName:ar,backendName:`cpu`,kernelFunc:TT};function DT(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,segmentIds:a}=t,{numSegments:o}=r;G(i,`unsortedSegmentSum`);let s=i.shape.length,c=a.shape.length,l=[],u=[],d=s-c,f=a;for(let e=0;e<d;++e){let t=DS({inputs:{input:f},backend:n,attrs:{dim:e+1}});f=t,u.push(t)}for(let e=0;e<o;++e){let t=Vr(e,`int32`),r=n.makeTensorInfo([],`int32`,t),a=Yv({inputs:{a:r,b:f},backend:n}),o=Pv({inputs:{x:a},backend:n,attrs:{dtype:`float32`}}),s=Cy({inputs:{a:o,b:i},backend:n}),c=mS({inputs:{x:s},backend:n,attrs:{axis:0,keepDims:!1}});l.push(c),u.push(r),u.push(a),u.push(o),u.push(s),u.push(c)}let p=XC({inputs:l,backend:n,attrs:{axis:0}});return u.forEach(e=>n.disposeIntermediateTensorInfo(e)),p}var OT=[Ub,wv,Wb,Gb,Bv,qb,Yb,Zb,$b,tx,nx,rx,ix,ax,ox,fx,mx,gx,vx,Vb,bx,Sx,wx,Ex,Fv,Kv,Dx,Dv,Ox,Mx,Px,Ix,Rx,Bx,Hx,Wx,Gx,Kx,Jx,Xx,Qx,eS,nS,iS,oS,cS,uS,dS,fS,pS,_S,Ob,yS,Xv,ES,$v,OS,ty,LS,zS,VS,ry,HS,WS,KS,JS,XS,sy,ly,Av,QS,Ax,$S,eC,tC,Ab,dy,py,rC,gy,iC,aC,oC,sC,lC,dC,pC,yy,hC,_C,yC,xC,CC,TC,DC,xy,kC,AC,PC,wy,Dy,LC,BC,UC,ky,GC,YC,ZC,$C,ew,Nb,Fy,nw,iw,ow,Mv,AS,sw,Fb,Lb,zb,lw,dw,pw,hw,_w,vw,yw,Zy,xw,Ew,Ow,jw,tb,Mw,Nw,Pw,ib,MC,Iw,Rw,Bw,Hw,Ww,Kw,Jw,Xw,lb,Zw,db,Qw,eT,nT,iT,oT,bb,hS,sT,cT,uT,fT,mT,My,wT,ET,{kernelName:or,backendName:`cpu`,kernelFunc:DT},qC];for(let e of OT)xr(e);var kT={},AT={alpha:!1,antialias:!1,premultipliedAlpha:!1,preserveDrawingBuffer:!1,depth:!1,stencil:!1,failIfMajorPerformanceCaveat:!0};function jT(e,t){kT[e]=t}function MT(e,t){if(!(e in kT)||t!=null){let n=PT(e,t);if(n!==null)kT[e]=n;else return console.log(`Could not get context for WebGL version`,e),null}let n=kT[e];return n==null||n.isContextLost()?(delete kT[e],MT(e)):(n.disable(n.DEPTH_TEST),n.disable(n.STENCIL_TEST),n.disable(n.BLEND),n.disable(n.DITHER),n.disable(n.POLYGON_OFFSET_FILL),n.disable(n.SAMPLE_COVERAGE),n.enable(n.SCISSOR_TEST),n.enable(n.CULL_FACE),n.cullFace(n.BACK),kT[e])}function NT(e){if(typeof OffscreenCanvas<`u`&&e===2)return new OffscreenCanvas(300,150);if(typeof document<`u`)return document.createElement(`canvas`);throw Error(`Cannot create a canvas in this context`)}function PT(e,t){if(e!==1&&e!==2)throw Error(`Cannot get WebGL rendering context, WebGL is disabled.`);let n=t??NT(e);return n.addEventListener(`webglcontextlost`,t=>{t.preventDefault(),delete kT[e]},!1),N().getBool(`SOFTWARE_WEBGL_ENABLED`)&&(AT.failIfMajorPerformanceCaveat=!1),e===1?n.getContext(`webgl`,AT)||n.getContext(`experimental-webgl`,AT):n.getContext(`webgl2`,AT)}var FT;(function(e){e[e.DENSE=0]=`DENSE`,e[e.SHARED_BATCH=1]=`SHARED_BATCH`})(FT||={});var IT;(function(e){e[e.RENDER=0]=`RENDER`,e[e.UPLOAD=1]=`UPLOAD`,e[e.PIXELS=2]=`PIXELS`,e[e.DOWNLOAD=3]=`DOWNLOAD`})(IT||={});var LT;(function(e){e[e.UNPACKED_FLOAT16=0]=`UNPACKED_FLOAT16`,e[e.UNPACKED_FLOAT32=1]=`UNPACKED_FLOAT32`,e[e.PACKED_4X1_UNSIGNED_BYTE=2]=`PACKED_4X1_UNSIGNED_BYTE`,e[e.PACKED_2X2_FLOAT32=3]=`PACKED_2X2_FLOAT32`,e[e.PACKED_2X2_FLOAT16=4]=`PACKED_2X2_FLOAT16`})(LT||={});function RT(e,t){return[t,e]}function zT(e,t){return e*t}function BT(e){let t=g(e);return y(Math.ceil(t/4))}function VT(e,t){return[Math.max(1,Math.ceil(t/2)),Math.max(1,Math.ceil(e/2))]}function HT(e,t){let[n,r]=VT(e,t);return n*r*4}function UT(e,t){let n=e,r,i,a,o,s,c,l,u,d,f;return N().getNumber(`WEBGL_VERSION`)===2?(r=n.R32F,i=n.R16F,a=n.RGBA16F,o=n.RGBA32F,s=n.RED,l=4,u=1,d=n.HALF_FLOAT,f=n.FLOAT,c=n.RGBA8):(r=e.RGBA,i=e.RGBA,a=e.RGBA,o=n.RGBA,s=e.RGBA,l=4,u=4,d=t==null?null:t.HALF_FLOAT_OES,f=e.FLOAT,c=e.RGBA),{internalFormatFloat:r,internalFormatHalfFloat:i,internalFormatPackedHalfFloat:a,internalFormatPackedFloat:o,textureFormatFloat:s,downloadTextureFormat:c,downloadUnpackNumChannels:l,defaultNumChannels:u,textureTypeHalfFloat:d,textureTypeFloat:f}}function J(e,t){let n=t();return N().getBool(`DEBUG`)&&WT(e),n}function WT(e){let t=e.getError();if(t!==e.NO_ERROR)throw Error(`WebGL Error: `+JT(e,t))}var GT=5.96e-8,KT=65504;function qT(e){return!!(N().getBool(`WEBGL_RENDER_FLOAT32_ENABLED`)||e===0||GT<Math.abs(e)&&Math.abs(e)<KT)}function JT(e,t){switch(t){case e.NO_ERROR:return`NO_ERROR`;case e.INVALID_ENUM:return`INVALID_ENUM`;case e.INVALID_VALUE:return`INVALID_VALUE`;case e.INVALID_OPERATION:return`INVALID_OPERATION`;case e.INVALID_FRAMEBUFFER_OPERATION:return`INVALID_FRAMEBUFFER_OPERATION`;case e.OUT_OF_MEMORY:return`OUT_OF_MEMORY`;case e.CONTEXT_LOST_WEBGL:return`CONTEXT_LOST_WEBGL`;default:return`Unknown error code ${t}`}}function YT(e,t){return _E(e,()=>e.getExtension(t),`Extension "`+t+`" not supported on this browser.`)}function XT(e,t){let n=_E(e,()=>e.createShader(e.VERTEX_SHADER),`Unable to create vertex WebGLShader.`);if(J(e,()=>e.shaderSource(n,t)),J(e,()=>e.compileShader(n)),e.getShaderParameter(n,e.COMPILE_STATUS)===!1)throw console.log(e.getShaderInfoLog(n)),Error(`Failed to compile vertex shader.`);return n}function ZT(e,t){let n=_E(e,()=>e.createShader(e.FRAGMENT_SHADER),`Unable to create fragment WebGLShader.`);if(J(e,()=>e.shaderSource(n,t)),J(e,()=>e.compileShader(n)),N().get(`ENGINE_COMPILE_ONLY`))return n;if(e.getShaderParameter(n,e.COMPILE_STATUS)===!1)throw $T(t,e.getShaderInfoLog(n)),Error(`Failed to compile fragment shader.`);return n}var QT=/ERROR: [0-9]+:([0-9]+):/g;function $T(e,t){let n=QT.exec(t);if(n==null){console.log(`Couldn't parse line number in error: ${t}`),console.log(e);return}let r=+n[1],i=e.split(`
`),a=i.length.toString().length+2,o=i.map((e,t)=>b((t+1).toString(),a)+e),s=0;for(let e=0;e<o.length;e++)s=Math.max(o[e].length,s);let c=o.slice(0,r-1),l=o.slice(r-1,r),u=o.slice(r);console.log(c.join(`
`)),console.log(t.split(`
`)[0]),console.log(`%c ${b(l[0],s)}`,`border:1px solid red; background-color:#e3d2d2; color:#a61717`),console.log(u.join(`
`))}function eE(e){return _E(e,()=>e.createProgram(),`Unable to create WebGLProgram.`)}function tE(e,t){if(J(e,()=>e.linkProgram(t)),!N().get(`ENGINE_COMPILE_ONLY`)&&e.getProgramParameter(t,e.LINK_STATUS)===!1)throw console.log(e.getProgramInfoLog(t)),Error(`Failed to link vertex and fragment shaders.`)}function nE(e,t){if(J(e,()=>e.validateProgram(t)),e.getProgramParameter(t,e.VALIDATE_STATUS)===!1)throw console.log(e.getProgramInfoLog(t)),Error(`Shader program validation failed.`)}function rE(e,t){let n=_E(e,()=>e.createBuffer(),`Unable to create WebGLBuffer`);return J(e,()=>e.bindBuffer(e.ARRAY_BUFFER,n)),J(e,()=>e.bufferData(e.ARRAY_BUFFER,t,e.STATIC_DRAW)),n}function iE(e,t){let n=_E(e,()=>e.createBuffer(),`Unable to create WebGLBuffer`);return J(e,()=>e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,n)),J(e,()=>e.bufferData(e.ELEMENT_ARRAY_BUFFER,t,e.STATIC_DRAW)),n}function aE(e){return _E(e,()=>e.createTexture(),`Unable to create WebGLTexture.`)}function oE(e,t){let n=N().getNumber(`WEBGL_MAX_TEXTURE_SIZE`);if(e<=0||t<=0){let n=`[${e}x${t}]`;throw Error(`Requested texture size `+n+` is invalid.`)}if(e>n||t>n){let r=`[${e}x${t}]`,i=`[${n}x${n}]`;throw Error(`Requested texture size `+r+` greater than WebGL maximum on this browser / GPU `+i+`.`)}}function sE(e){return _E(e,()=>e.createFramebuffer(),`Unable to create WebGLFramebuffer.`)}function cE(e,t,n,r,i,a,o){let s=e.getAttribLocation(t,n);return s!==-1&&(J(e,()=>e.bindBuffer(e.ARRAY_BUFFER,r)),J(e,()=>e.vertexAttribPointer(s,i,e.FLOAT,!1,a,o)),J(e,()=>e.enableVertexAttribArray(s)),!0)}function lE(e,t,n){vE(e,n),J(e,()=>e.activeTexture(e.TEXTURE0+n)),J(e,()=>e.bindTexture(e.TEXTURE_2D,t))}function uE(e,t,n){return _E(e,()=>e.getUniformLocation(t,n),`uniform "`+n+`" not present in program.`)}function dE(e,t,n){return e.getUniformLocation(t,n)}function fE(e,t,n,r){J(e,()=>lE(e,t,r)),J(e,()=>e.uniform1i(n,r))}function pE(e,t,n){J(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,n)),J(e,()=>e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,t,0))}function mE(e,t){J(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,t)),J(e,()=>e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,null,0))}function hE(e){let t=e.checkFramebufferStatus(e.FRAMEBUFFER);if(t!==e.FRAMEBUFFER_COMPLETE)throw Error(`Error binding framebuffer: `+gE(e,t))}function gE(e,t){switch(t){case e.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:return`FRAMEBUFFER_INCOMPLETE_ATTACHMENT`;case e.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:return`FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT`;case e.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:return`FRAMEBUFFER_INCOMPLETE_DIMENSIONS`;case e.FRAMEBUFFER_UNSUPPORTED:return`FRAMEBUFFER_UNSUPPORTED`;default:return`unknown error ${t}`}}function _E(e,t,n){let r=J(e,()=>t());if(r==null)throw Error(n);return r}function vE(e,t){let n=e.MAX_COMBINED_TEXTURE_IMAGE_UNITS-1,r=t+e.TEXTURE0;if(r<e.TEXTURE0||r>n){let e=`[gl.TEXTURE0, gl.TEXTURE${n}]`;throw Error(`textureUnit must be in ${e}.`)}}function yE(e,t=2){return g(e.slice(0,e.length-t))}function bE(e){if(e.length===0)throw Error(`Cannot get rows and columns of an empty shape array.`);return[e.length>1?e[e.length-2]:1,e[e.length-1]]}function xE(e){let t=[1,1,1];return e.length===0||e.length===1&&e[0]===1||(t=[yE(e),...bE(e)]),t}function SE(e,t=!1){let n=N().getNumber(`WEBGL_MAX_TEXTURE_SIZE`),r=N().getNumber(`WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE`);r===1/0&&N().getBool(`WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE`)&&(r=n/2),t&&(n*=2,r*=2,e=e.map((t,n)=>n>=e.length-2?l(e[n]):e[n]),e.length===1&&(e=[2,e[0]])),e.length!==2&&(e=w(e).newShape);let i=g(e),a=null;e.length<=1&&i<=n?a=[1,i]:e.length===2&&e[0]<=n&&e[1]<=n?a=e:e.length===3&&e[0]*e[1]<=n&&e[2]<=n?a=[e[0]*e[1],e[2]]:e.length===3&&e[0]<=n&&e[1]*e[2]<=n?a=[e[0],e[1]*e[2]]:e.length===4&&e[0]*e[1]*e[2]<=n&&e[3]<=n?a=[e[0]*e[1]*e[2],e[3]]:e.length===4&&e[0]<=n&&e[1]*e[2]*e[3]<=n&&(a=[e[0],e[1]*e[2]*e[3]]);let o=a!=null&&Math.max(...a)>r&&Math.min(...a)<=(t?2:1)&&Math.min(...a)>0;if(a==null||o){if(t){let t=yE(e),n=2,r=2;e.length&&([n,r]=bE(e)),i=n/2*t*(r/2),a=y(i).map(e=>e*2)}else a=y(i)}return a}function CE(e){return e%2==0}function wE(e,t){if(e=e.slice(-2),t=t.slice(-2),_(e,t)||!e.length||!t.length||e[0]===0||e[1]===0||t[0]===0||t[1]===0)return!0;if(e.length!==t.length){let n=e.slice(-1)[0],r=t.slice(-1)[0];if(n===r||CE(n)&&CE(r)&&(e[0]===1||t[0]===1))return!0}return e[1]===t[1]&&CE(e[0])&&CE(t[0])}var TE,EE;function DE(e){if(TE==null){let t=MT(e);TE=t.getParameter(t.MAX_TEXTURE_SIZE)}return TE}function OE(e){if(EE==null){let t=MT(e);EE=t.getParameter(t.MAX_TEXTURE_IMAGE_UNITS)}return Math.min(16,EE)}function kE(e){if(e===0)return 0;let t,n=MT(e);return t=AE(n,`EXT_disjoint_timer_query_webgl2`)&&e===2?2:+!!AE(n,`EXT_disjoint_timer_query`),t}function AE(e,t){return e.getExtension(t)!=null}function jE(e){try{if(MT(e)!=null)return!0}catch(e){return console.log(`Error when getting WebGL context: `,e),!1}return!1}function ME(e){if(e===0)return!1;let t=MT(e);if(e===1){if(!AE(t,`OES_texture_float`))return!1}else if(!AE(t,`EXT_color_buffer_float`))return!1;return PE(t)}function NE(e){if(e===0)return!1;let t=MT(e);if(e===1){if(!AE(t,`OES_texture_float`)||!AE(t,`WEBGL_color_buffer_float`))return!1}else{if(AE(t,`EXT_color_buffer_float`))return PE(t);let e=`EXT_color_buffer_half_float`;return AE(t,e)?FE(t,t.getExtension(e)):!1}return PE(t)}function PE(e){let t=UT(e),n=e.createTexture();e.bindTexture(e.TEXTURE_2D,n),e.texImage2D(e.TEXTURE_2D,0,t.internalFormatFloat,1,1,0,t.textureFormatFloat,t.textureTypeFloat,null);let r=e.createFramebuffer();e.bindFramebuffer(e.FRAMEBUFFER,r),e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,n,0);let i=e.checkFramebufferStatus(e.FRAMEBUFFER)===e.FRAMEBUFFER_COMPLETE;return e.bindTexture(e.TEXTURE_2D,null),e.bindFramebuffer(e.FRAMEBUFFER,null),e.deleteTexture(n),e.deleteFramebuffer(r),i}function FE(e,t){let n=UT(e,t),r=e.createTexture();e.bindTexture(e.TEXTURE_2D,r),e.texImage2D(e.TEXTURE_2D,0,n.internalFormatHalfFloat,1,1,0,n.textureFormatFloat,n.textureTypeHalfFloat,null);let i=e.createFramebuffer();e.bindFramebuffer(e.FRAMEBUFFER,i),e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,r,0);let a=e.checkFramebufferStatus(e.FRAMEBUFFER)===e.FRAMEBUFFER_COMPLETE;return e.bindTexture(e.TEXTURE_2D,null),e.bindFramebuffer(e.FRAMEBUFFER,null),e.deleteTexture(r),e.deleteFramebuffer(i),a}function IE(e){return e===2&&MT(e).fenceSync!=null}function LE(e,t){Array.isArray(e)||(e=[e]),e.forEach(e=>{e!=null&&f(e.dtype!==`complex64`,()=>`${t} does not support complex64 tensors in the WebGL backend.`)})}var Y=N();Y.registerFlag(`HAS_WEBGL`,()=>Y.getNumber(`WEBGL_VERSION`)>0),Y.registerFlag(`WEBGL_VERSION`,()=>jE(2)?2:+!!jE(1)),Y.registerFlag(`WEBGL_CHECK_NUMERICAL_PROBLEMS`,()=>!1),Y.registerFlag(`WEBGL_BUFFER_SUPPORTED`,()=>Y.get(`WEBGL_VERSION`)===2),Y.registerFlag(`WEBGL_CPU_FORWARD`,()=>!0),Y.registerFlag(`WEBGL_FORCE_F16_TEXTURES`,()=>!1),Y.registerFlag(`WEBGL_PACK`,()=>Y.getBool(`HAS_WEBGL`)),Y.registerFlag(`WEBGL_PACK_NORMALIZATION`,()=>Y.getBool(`WEBGL_PACK`)),Y.registerFlag(`WEBGL_PACK_CLIP`,()=>Y.getBool(`WEBGL_PACK`)),Y.registerFlag(`WEBGL_PACK_DEPTHWISECONV`,()=>Y.getBool(`WEBGL_PACK`)),Y.registerFlag(`WEBGL_PACK_BINARY_OPERATIONS`,()=>Y.getBool(`WEBGL_PACK`)),Y.registerFlag(`WEBGL_PACK_UNARY_OPERATIONS`,()=>Y.getBool(`WEBGL_PACK`)),Y.registerFlag(`WEBGL_PACK_ARRAY_OPERATIONS`,()=>Y.getBool(`WEBGL_PACK`)),Y.registerFlag(`WEBGL_PACK_IMAGE_OPERATIONS`,()=>Y.getBool(`WEBGL_PACK`)),Y.registerFlag(`WEBGL_PACK_REDUCE`,()=>Y.getBool(`WEBGL_PACK`)),Y.registerFlag(`WEBGL_LAZILY_UNPACK`,()=>Y.getBool(`WEBGL_PACK`)),Y.registerFlag(`WEBGL_CONV_IM2COL`,()=>Y.getBool(`WEBGL_PACK`)),Y.registerFlag(`WEBGL_MAX_TEXTURE_SIZE`,()=>DE(Y.getNumber(`WEBGL_VERSION`))),Y.registerFlag(`WEBGL_MAX_TEXTURES_IN_SHADER`,()=>OE(Y.getNumber(`WEBGL_VERSION`))),Y.registerFlag(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`,()=>{let e=Y.getNumber(`WEBGL_VERSION`);return e===0?0:kE(e)}),Y.registerFlag(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`,()=>Y.getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)>0&&!Fi()),Y.registerFlag(`WEBGL_RENDER_FLOAT32_CAPABLE`,()=>ME(Y.getNumber(`WEBGL_VERSION`))),Y.registerFlag(`WEBGL_RENDER_FLOAT32_ENABLED`,()=>!Y.getBool(`WEBGL_FORCE_F16_TEXTURES`)&&Y.getBool(`WEBGL_RENDER_FLOAT32_CAPABLE`)),Y.registerFlag(`WEBGL_DOWNLOAD_FLOAT_ENABLED`,()=>NE(Y.getNumber(`WEBGL_VERSION`))),Y.registerFlag(`WEBGL_FENCE_API_ENABLED`,()=>IE(Y.getNumber(`WEBGL_VERSION`))),Y.registerFlag(`WEBGL_SIZE_UPLOAD_UNIFORM`,()=>Y.getBool(`WEBGL_RENDER_FLOAT32_ENABLED`)?4:0),Y.registerFlag(`WEBGL_DELETE_TEXTURE_THRESHOLD`,()=>-1,e=>{if(e<0&&e!==-1)throw Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be -1 (indicating never delete) or at least 0, but got ${e}.`)}),Y.registerFlag(`WEBGL_FLUSH_THRESHOLD`,()=>Fi()?1:-1,e=>{if(e<0&&e!==-1)throw Error(`WEBGL_FLUSH_THRESHOLD must be -1 (indicating never manual flush) or at least 0, but got ${e}.`)}),Y.registerFlag(`CPU_HANDOFF_SIZE_THRESHOLD`,()=>128),Y.registerFlag(`WEBGL_USE_SHAPES_UNIFORMS`,()=>!1),Y.registerFlag(`TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD`,()=>1e5),Y.registerFlag(`TOPK_K_CPU_HANDOFF_THRESHOLD`,()=>128),Y.registerFlag(`WEBGL_EXP_CONV`,()=>!1),Y.registerFlag(`SOFTWARE_WEBGL_ENABLED`,()=>Y.getBool(`IS_TEST`)),Y.registerFlag(`WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE`,()=>1/0),Y.registerFlag(`WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE`,()=>!1),Y.registerFlag(`WEBGL2_ISNAN_CUSTOM`,()=>!1);function RE(){let e,t,n,r,i,a,o,s,c,l;return N().getNumber(`WEBGL_VERSION`)===2?(e=`#version 300 es`,t=`in`,n=`out`,r=`in`,i=`texture`,a=`outputColor`,o=`out vec4 outputColor;`,s=N().getBool(`WEBGL2_ISNAN_CUSTOM`)?`
      bool isnan_custom(float val) {
        uint floatToUint = floatBitsToUint(val);
        return (floatToUint & 0x7fffffffu) > 0x7f800000u;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    `:``,c=``,l=`
      #define round(value) newRound(value)
      int newRound(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 newRound(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `):(e=``,t=`attribute`,n=`varying`,r=`varying`,i=`texture2D`,a=`gl_FragColor`,o=``,s=`
      #define isnan(value) isnan_custom(value)
      bool isnan_custom(float val) {
        return (val > 0. || val < 1. || val == 0.) ? false : true;
      }
      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan(val.x), isnan(val.y), isnan(val.z), isnan(val.w));
      }
    `,c=`
      uniform float INFINITY;

      bool isinf(float val) {
        return abs(val) == INFINITY;
      }
      bvec4 isinf(vec4 val) {
        return equal(abs(val), vec4(INFINITY));
      }
    `,l=`
      int round(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 round(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `),{version:e,attribute:t,varyingVs:n,varyingFs:r,texture2D:i,output:a,defineOutput:o,defineSpecialNaN:s,defineSpecialInf:c,defineRound:l}}function zE(e,t,n=`index`){let r=j(t);return r.map((t,i)=>`${`int ${e[i]} = ${n} / ${t}`}; ${i===r.length-1?`int ${e[i+1]} = ${n} - ${e[i]} * ${t}`:`index -= ${e[i]} * ${t}`};`).join(``)}function BE(e,t,n=`index`){let r=j(t);return r.map((t,i)=>`${`int ${e[i]} = ${n} / outShapeStrides[${i}]`}; ${i===r.length-1?`int ${e[i+1]} = ${n} - ${e[i]} * outShapeStrides[${i}]`:`index -= ${e[i]} * outShapeStrides[${i}]`};`).join(``)}function VE(e,t){let n=e.length,r=e.map(e=>`${t}[${e}]`),i=Array(n-1);i[n-2]=r[n-1];for(let e=n-3;e>=0;--e)i[e]=`(${i[e+1]} * ${r[e+1]})`;return i}function HE(e,t,n=`index`){let r=VE(e.map((e,t)=>t),t);return r.map((t,i)=>`${`int ${e[i]} = ${n} / ${r[i]}`}; ${i===r.length-1?`int ${e[i+1]} = ${n} - ${e[i]} * ${r[i]}`:`index -= ${e[i]} * ${r[i]}`};`).join(``)}function UE(e){let t=j(e).map(e=>e.toString());return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * ${t[0]} + coords.y * ${t[1]} + coords.z;
  }
`}function WE(){return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * outShapeStrides[0] + coords.y * outShapeStrides[1] + coords.z;
  }
`}var GE=`
  const float FLOAT_MAX = 1.70141184e38;
  const float FLOAT_MIN = 1.17549435e-38;

  lowp vec4 encode_float(highp float v) {
    if (isnan(v)) {
      return vec4(255, 255, 255, 255);
    }

    highp float av = abs(v);

    if(av < FLOAT_MIN) {
      return vec4(0.0, 0.0, 0.0, 0.0);
    } else if(v > FLOAT_MAX) {
      return vec4(0.0, 0.0, 128.0, 127.0) / 255.0;
    } else if(v < -FLOAT_MAX) {
      return vec4(0.0, 0.0,  128.0, 255.0) / 255.0;
    }

    highp vec4 c = vec4(0,0,0,0);

    highp float e = floor(log2(av));
    highp float m = exp2(fract(log2(av))) - 1.0;

    c[2] = floor(128.0 * m);
    m -= c[2] / 128.0;
    c[1] = floor(32768.0 * m);
    m -= c[1] / 32768.0;
    c[0] = floor(8388608.0 * m);

    highp float ebias = e + 127.0;
    c[3] = floor(ebias / 2.0);
    ebias -= c[3] * 2.0;
    c[2] += floor(ebias) * 128.0;

    c[3] += 128.0 * step(0.0, -v);

    return c / 255.0;
  }
`,{getBroadcastDims:KE}=vv;function qE(e,t,n){let r=[];if(e.forEach(e=>{let t=g(e.shapeInfo.logicalShape);if(e.shapeInfo.isUniform?r.push(`uniform float ${e.name}${t>1?`[${t}]`:``};`):(r.push(`uniform sampler2D ${e.name};`),r.push(`uniform int offset${e.name};`)),n.enableShapeUniforms){let{uniformShape:t}=PD(n.packedInputs,e.shapeInfo.logicalShape,e.shapeInfo.texShape);switch(t.length){case 1:r.push(`uniform int ${e.name}Shape;`);break;case 2:r.push(`uniform ivec2 ${e.name}Shape;`);break;case 3:r.push(`uniform ivec3 ${e.name}Shape;`);break;case 4:r.push(`uniform ivec4 ${e.name}Shape;`)}r.push(`uniform ivec2 ${e.name}TexShape;`)}}),n.enableShapeUniforms){switch(t.logicalShape.length){case 1:r.push(`uniform int outShape;`);break;case 2:r.push(`uniform ivec2 outShape;`),r.push(`uniform int outShapeStrides;`);break;case 3:r.push(`uniform ivec3 outShape;`),r.push(`uniform ivec2 outShapeStrides;`);break;case 4:r.push(`uniform ivec4 outShape;`),r.push(`uniform ivec3 outShapeStrides;`)}r.push(`uniform ivec2 outTexShape;`)}n.customUniforms&&n.customUniforms.forEach(e=>{r.push(`uniform ${e.type} ${e.name}${e.arrayIndex?`[${e.arrayIndex}]`:``};`)});let i=r.join(`
`),a=e.map(e=>XE(e,t,n.packedInputs,n.enableShapeUniforms)).join(`
`),o=t.texShape,s=RE(),c=$E(s),l,u,d=nD(s);return t.isPacked?(l=ZE(t.logicalShape,o,n.enableShapeUniforms),u=tD(s)):(l=QE(t.logicalShape,o,n.enableShapeUniforms),u=eD(s)),n.packedInputs&&(d+=oD),[d,c,u,i,l,a,n.userCode].join(`
`)}function JE(e,t=!1){let n=e.shapeInfo.logicalShape;switch(n.length){case 0:return bD(e,t);case 1:return SD(e,t);case 2:return wD(e,t);case 3:return ED(e,t);case 4:return OD(e,t);case 5:return kD(e);case 6:return AD(e);default:throw Error(`${n.length}-D input sampling is not yet supported`)}}function YE(e,t){switch(e.shapeInfo.logicalShape.length){case 0:return yD(e);case 1:return xD(e,t);case 2:return CD(e,t);case 3:return TD(e,t);default:return DD(e,t)}}function XE(e,t,n=!1,r){let i=``;i+=n?YE(e,r):JE(e,r);let a=e.shapeInfo.logicalShape,o=t.logicalShape;return a.length<=o.length&&(i+=n?MD(e,t):ND(e,t)),i}function ZE(e,t,n){switch(e.length){case 0:return sD();case 1:return cD(e,t,n);case 2:return gD(e,t,n);case 3:return uD(e,t,n);default:return fD(e,t,n)}}function QE(e,t,n){switch(e.length){case 0:return sD();case 1:return lD(e,t,n);case 2:return _D(e,t,n);case 3:return dD(e,t,n);case 4:return pD(e,t,n);case 5:return mD(e,t);case 6:return hD(e,t);default:throw Error(`${e.length}-D output sampling is not yet supported`)}}function $E(e){return`
    float sampleTexture(sampler2D textureSampler, vec2 uv) {
      return ${e.texture2D}(textureSampler, uv).r;
    }
  `}function eD(e){return`
    void setOutput(float val) {
      ${e.output} = vec4(val, 0, 0, 0);
    }
  `}function tD(e){return`
    void setOutput(vec4 val) {
      ${e.output} = val;
    }
  `}function nD(e){return`${e.version}
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    ${e.varyingFs} vec2 resultUV;
    ${e.defineOutput}
    const vec2 halfCR = vec2(0.5, 0.5);

    struct ivec5
    {
      int x;
      int y;
      int z;
      int w;
      int u;
    };

    struct ivec6
    {
      int x;
      int y;
      int z;
      int w;
      int u;
      int v;
    };

    uniform float NAN;
    ${e.defineSpecialNaN}
    ${e.defineSpecialInf}
    ${e.defineRound}

    int imod(int x, int y) {
      return x - y * (x / y);
    }

    int idiv(int a, int b, float sign) {
      int res = a / b;
      int mod = imod(a, b);
      if (sign < 0. && mod != 0) {
        res -= 1;
      }
      return res;
    }

    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    #define HASHSCALE1 443.8975
    float random(float seed){
      vec2 p = resultUV * seed;
      vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
      p3 += dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${rD}
    ${iD}
    ${aD}
  `}var rD=`
vec2 uvFromFlat(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 2;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,iD=`
vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,aD=`
vec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,oD=`
  float getChannel(vec4 frag, vec2 innerDims) {
    vec2 modCoord = mod(innerDims, 2.);
    return modCoord.x == 0. ?
      (modCoord.y == 0. ? frag.r : frag.g) :
      (modCoord.y == 0. ? frag.b : frag.a);
  }
  float getChannel(vec4 frag, int dim) {
    float modCoord = mod(float(dim), 2.);
    return modCoord == 0. ? frag.r : frag.g;
  }
`;function sD(){return`
    int getOutputCoords() {
      return 0;
    }
  `}function cD(e,t,n){let r=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)];return r[0]===1?n?`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ceil(float(outTexShape[1]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ${r[1]}.0);
      }
    `:r[1]===1?n?`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ceil(float(outTexShape[0]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ${r[0]}.0);
      }
    `:n?`
    int getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      return 2 * (resTexRC.x * packedTexShape[1] + resTexRC.y);
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${r[0]}, ${r[1]}));
      return 2 * (resTexRC.x * ${r[1]} + resTexRC.y);
    }
  `}function lD(e,t,n){return t[0]===1?n?`
      int getOutputCoords() {
        return int(resultUV.x * float(outTexShape[1]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.x * ${t[1]}.0);
      }
    `:t[1]===1?n?`
      int getOutputCoords() {
        return int(resultUV.y * float(outTexShape[0]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.y * ${t[0]}.0);
      }
    `:n?`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      return resTexRC.x * outTexShape[1] + resTexRC.y;
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      return resTexRC.x * ${t[1]} + resTexRC.y;
    }
  `}function uD(e,t,n){if(n)return`
    ivec3 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec3(b, r, c);
    }
  `;let r=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)],i=Math.ceil(e[2]/2),a=i*Math.ceil(e[1]/2);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${r[0]}, ${r[1]}));
      int index = resTexRC.x * ${r[1]} + resTexRC.y;

      int b = index / ${a};
      index -= b * ${a};

      int r = 2 * (index / ${i});
      int c = imod(index, ${i}) * 2;

      return ivec3(b, r, c);
    }
  `}function dD(e,t,n){if(n)return`
  ivec3 getOutputCoords() {
    ivec2 resTexRC = ivec2(resultUV.yx *
                           vec2(outTexShape[0], outTexShape[1]));
    int index = resTexRC.x * outTexShape[1] + resTexRC.y;
    ${BE([`r`,`c`,`d`],e)}
    return ivec3(r, c, d);
  }
`;let r=zE([`r`,`c`,`d`],e);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      ${r}
      return ivec3(r, c, d);
    }
  `}function fD(e,t,n){if(n)return`
    ivec4 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int texelsInLogicalRow = int(ceil(float(outShape[3]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatchN = texelsInBatch * outShape[1];

      int b2 = index / texelsInBatchN;
      index -= b2 * texelsInBatchN;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec4(b2, b, r, c);
    }
  `;let r=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)],i=Math.ceil(e[e.length-1]/2),a=i*Math.ceil(e[e.length-2]/2),o=a,s=``,c=`b, r, c`;for(let t=2;t<e.length-1;t++)o*=e[e.length-t-1],s=`
      int b${t} = index / ${o};
      index -= b${t} * ${o};
    `+s,c=`b${t}, `+c;return`
    ivec${e.length} getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${r[0]}, ${r[1]}));
      int index = resTexRC.x * ${r[1]} + resTexRC.y;

      ${s}

      int b = index / ${a};
      index -= b * ${a};

      int r = 2 * (index / ${i});
      int c = imod(index, ${i}) * 2;

      return ivec${e.length}(${c});
    }
  `}function pD(e,t,n){if(n)return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      ${BE([`r`,`c`,`d`,`d2`],e)}
      return ivec4(r, c, d, d2);
    }
  `;let r=zE([`r`,`c`,`d`,`d2`],e);return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      ${r}
      return ivec4(r, c, d, d2);
    }
  `}function mD(e,t){let n=zE([`r`,`c`,`d`,`d2`,`d3`],e);return`
    ivec5 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${t[0]},
                             ${t[1]}));

      int index = resTexRC.x * ${t[1]} + resTexRC.y;

      ${n}

      ivec5 outShape = ivec5(r, c, d, d2, d3);
      return outShape;
    }
  `}function hD(e,t){let n=zE([`r`,`c`,`d`,`d2`,`d3`,`d4`],e);return`
    ivec6 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;

      ${n}

      ivec6 result = ivec6(r, c, d, d2, d3, d4);
      return result;
    }
  `}function gD(e,t,n){let r=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)];if(_(e,t))return n?`
      ivec2 getOutputCoords() {
        ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
        return 2 * ivec2(resultUV.yx * vec2(packedTexShape[0], packedTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return 2 * ivec2(resultUV.yx * vec2(${r[0]}, ${r[1]}));
      }
    `;let i=Math.ceil(e[1]/2);return n?`
    ivec2 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));

      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;
      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${r[0]}, ${r[1]}));

      int index = resTexRC.x * ${r[1]} + resTexRC.y;
      int r = 2 * (index / ${i});
      int c = imod(index, ${i}) * 2;

      return ivec2(r, c);
    }
  `}function _D(e,t,n){return _(e,t)?n?`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(outTexShape[0], outTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(${t[0]}, ${t[1]}));
      }
    `:e[1]===1?n?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(index, 0);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${t[0]}, ${t[1]}));
        int index = resTexRC.x * ${t[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `:e[0]===1?n?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(0, index);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${t[0]}, ${t[1]}));
        int index = resTexRC.x * ${t[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `:n?`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      int r = index / outShape[1];
      int c = index - r * outShape[1];
      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      int r = index / ${e[1]};
      int c = index - r * ${e[1]};
      return ivec2(r, c);
    }
  `}function vD(e){return`offset${e}`}function yD(e){let t=e.name;return`
    vec4 ${`get`+t.charAt(0).toUpperCase()+t.slice(1)}() {
      return ${RE().texture2D}(${t}, halfCR);
    }
  `}function bD(e,t){let n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1);if(e.shapeInfo.isUniform)return`float ${r}() {return ${n};}`;let[i,a]=e.shapeInfo.texShape;if(i===1&&a===1)return`
      float ${r}() {
        return sampleTexture(${n}, halfCR);
      }
    `;let o=vD(n);if(t)return`
    float ${r}() {
      vec2 uv = uvFromFlat(${n}TexShape[0], ${n}TexShape[1], ${o});
      return sampleTexture(${n}, uv);
    }
  `;let[s,c]=e.shapeInfo.texShape;return`
    float ${r}() {
      vec2 uv = uvFromFlat(${s}, ${c}, ${o});
      return sampleTexture(${n}, uv);
    }
  `}function xD(e,t){let n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1),i=e.shapeInfo.texShape,a=RE();if(t)return`
    vec4 ${r}(int index) {
      ivec2 packedTexShape = ivec2(ceil(float(${n}TexShape[0]) / 2.0), ceil(float(${n}TexShape[1]) / 2.0));
      vec2 uv = packedUVfrom1D(
        packedTexShape[0], packedTexShape[1], index);
      return ${a.texture2D}(${n}, uv);
    }
  `;let o=[Math.ceil(i[0]/2),Math.ceil(i[1]/2)];return`
    vec4 ${r}(int index) {
      vec2 uv = packedUVfrom1D(
        ${o[0]}, ${o[1]}, index);
      return ${a.texture2D}(${n}, uv);
    }
  `}function SD(e,t){let n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1);if(e.shapeInfo.isUniform)return`
      float ${r}(int index) {
        ${jD(e)}
      }
    `;let i=e.shapeInfo.texShape,a=i[0],o=i[1];if(o===1&&a===1)return`
      float ${r}(int index) {
        return sampleTexture(${n}, halfCR);
      }
    `;let s=vD(n);return o===1?t?`
      float ${r}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${s}) + 0.5) / float(${n}TexShape[0]));
        return sampleTexture(${n}, uv);
      }
    `:`
      float ${r}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${s}) + 0.5) / ${a}.0);
        return sampleTexture(${n}, uv);
      }
    `:a===1?t?`
      float ${r}(int index) {
        vec2 uv = vec2((float(index + ${s}) + 0.5) / float(${n}TexShape[1]), 0.5);
        return sampleTexture(${n}, uv);
      }
    `:`
      float ${r}(int index) {
        vec2 uv = vec2((float(index + ${s}) + 0.5) / ${o}.0, 0.5);
        return sampleTexture(${n}, uv);
      }
    `:t?`
    float ${r}(int index) {
      vec2 uv = uvFromFlat(${n}TexShape[0], ${n}TexShape[1], index + ${s});
      return sampleTexture(${n}, uv);
    }
  `:`
    float ${r}(int index) {
      vec2 uv = uvFromFlat(${a}, ${o}, index + ${s});
      return sampleTexture(${n}, uv);
    }
  `}function CD(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=e.shapeInfo.texShape,o=a[0],s=a[1],c=RE();if(a!=null&&_(n,a))return t?`
      vec4 ${i}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${r}TexShape[1], ${r}TexShape[0]);

        return ${c.texture2D}(${r}, uv);
      }
    `:`
      vec4 ${i}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${s}.0, ${o}.0);

        return ${c.texture2D}(${r}, uv);
      }
    `;if(t)return`
    vec4 ${i}(int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${r}TexShape[0]) / 2.0), ceil(float(${r}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${r}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom2D(valuesPerRow, packedTexShape[0], packedTexShape[1], row, col);
      return ${c.texture2D}(${r}, uv);
    }
  `;let l=[Math.ceil(a[0]/2),Math.ceil(a[1]/2)];return`
    vec4 ${i}(int row, int col) {
      vec2 uv = packedUVfrom2D(${Math.ceil(n[1]/2)}, ${l[0]}, ${l[1]}, row, col);
      return ${c.texture2D}(${r}, uv);
    }
  `}function wD(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=e.shapeInfo.texShape;if(a!=null&&_(n,a)){if(t)return`
      float ${i}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `;let e=a[0];return`
    float ${i}(int row, int col) {
      vec2 uv = (vec2(col, row) + halfCR) / vec2(${a[1]}.0, ${e}.0);
      return sampleTexture(${r}, uv);
    }
  `}let{newShape:o,keptDims:s}=w(n),c=o;if(c.length<n.length)return`
      ${JE(FD(e,c),t)}
      float ${i}(int row, int col) {
        return ${i}(${ID([`row`,`col`],s)});
      }
    `;if(e.shapeInfo.isUniform)return`
      float ${i}(int row, int col) {
        int index = round(dot(vec2(row, col), vec2(${n[1]}, 1)));
        ${jD(e)}
      }
    `;let l=a[0],u=a[1],d=vD(r);return u===1?t?`
      float ${i}(int row, int col) {
        float index = dot(vec3(row, col, ${d}), vec3(${r}Shape[1], 1, 1));
        vec2 uv = vec2(0.5, (index + 0.5) / float(${r}TexShape[0]));
        return sampleTexture(${r}, uv);
      }
    `:`
    float ${i}(int row, int col) {
      float index = dot(vec3(row, col, ${d}), vec3(${n[1]}, 1, 1));
      vec2 uv = vec2(0.5, (index + 0.5) / ${l}.0);
      return sampleTexture(${r}, uv);
    }
  `:l===1?t?`
      float ${i}(int row, int col) {
        float index = dot(vec3(row, col, ${d}), vec3(${r}Shape[1], 1, 1));
        vec2 uv = vec2((index + 0.5) / float(${r}TexShape[1]), 0.5);
        return sampleTexture(${r}, uv);
      }
    `:`
    float ${i}(int row, int col) {
      float index = dot(vec3(row, col, ${d}), vec3(${n[1]}, 1, 1));
      vec2 uv = vec2((index + 0.5) / ${u}.0, 0.5);
      return sampleTexture(${r}, uv);
    }
  `:t?`
      float ${i}(int row, int col) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${r}Shape[1] + col + ${d};
        vec2 uv = uvFromFlat(${r}TexShape[0], ${r}TexShape[1], index);
        return sampleTexture(${r}, uv);
      }
    `:`
  float ${i}(int row, int col) {
    // Explicitly use integer operations as dot() only works on floats.
    int index = row * ${n[1]} + col + ${d};
    vec2 uv = uvFromFlat(${l}, ${u}, index);
    return sampleTexture(${r}, uv);
  }
`}function TD(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=e.shapeInfo.texShape,o=[Math.ceil(a[0]/2),Math.ceil(a[1]/2)];if(n[0]===1)return`
        ${YE(FD(e,n.slice(1)),t)}
        vec4 ${i}(int b, int row, int col) {
          return ${i}(${ID([`b`,`row`,`col`],[1,2])});
        }
      `;let s=RE();if(t)return`
    vec4 ${i}(int b, int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${r}TexShape[0]) / 2.0), ceil(float(${r}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${r}Shape[2]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${r}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom3D(
        packedTexShape[0], packedTexShape[1], texelsInBatch, valuesPerRow, b, row, col);
      return ${s.texture2D}(${r}, uv);
    }
  `;let c=o[0],l=o[1],u=Math.ceil(n[2]/2);return`
    vec4 ${i}(int b, int row, int col) {
      vec2 uv = packedUVfrom3D(
        ${c}, ${l}, ${u*Math.ceil(n[1]/2)}, ${u}, b, row, col);
      return ${s.texture2D}(${r}, uv);
    }
  `}function ED(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=n[1]*n[2],o=n[2],{newShape:s,keptDims:c}=w(n),l=s;if(l.length<n.length)return`
        ${JE(FD(e,l),t)}
        float ${i}(int row, int col, int depth) {
          return ${i}(${ID([`row`,`col`,`depth`],c)});
        }
      `;if(e.shapeInfo.isUniform)return`
      float ${i}(int row, int col, int depth) {
        int index = round(dot(vec3(row, col, depth),
                          vec3(${a}, ${o}, 1)));
        ${jD(e)}
      }
    `;let u=e.shapeInfo.texShape,d=u[0],f=u[1],p=e.shapeInfo.flatOffset;if(f===a&&p==null)return t?`
      float ${i}(int row, int col, int depth) {
        int stride1 = ${r}Shape[2];
        float texR = float(row);
        float texC = dot(vec2(col, depth), vec2(stride1, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `:`
        float ${i}(int row, int col, int depth) {
          float texR = float(row);
          float texC = dot(vec2(col, depth), vec2(${o}, 1));
          vec2 uv = (vec2(texC, texR) + halfCR) /
                     vec2(${f}.0, ${d}.0);
          return sampleTexture(${r}, uv);
        }
      `;if(f===o&&p==null)return t?`
      float ${i}(int row, int col, int depth) {
        float texR = dot(vec2(row, col), vec2(${r}Shape[1], 1));
        float texC = float(depth);
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `:`
    float ${i}(int row, int col, int depth) {
      float texR = dot(vec2(row, col), vec2(${n[1]}, 1));
      float texC = float(depth);
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${f}.0, ${d}.0);
      return sampleTexture(${r}, uv);
    }
  `;let m=vD(r);return t?`
    float ${i}(int row, int col, int depth) {
      // Explicitly use integer operations as dot() only works on floats.
      int stride0 = ${r}Shape[1] * ${r}Shape[2];
      int stride1 = ${r}Shape[2];
      int index = row * stride0 + col * stride1 + depth + ${m};
      vec2 uv = uvFromFlat(${r}TexShape[0], ${r}TexShape[1], index);
      return sampleTexture(${r}, uv);
    }
    `:`
      float ${i}(int row, int col, int depth) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${a} + col * ${o} + depth + ${m};
        vec2 uv = uvFromFlat(${d}, ${f}, index);
        return sampleTexture(${r}, uv);
      }
  `}function DD(e,t){let n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1),i=RE();if(t)return`
    vec4 ${r}(int b2, int b, int row, int col) {
      int valuesPerRow = int(ceil(float(${n}Shape[3]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${n}Shape[2]) / 2.0));
      int index = b * texelsInBatch + (row / 2) * valuesPerRow + (col / 2);
      texelsInBatch *= ${n}Shape[1];
      index = b2 * texelsInBatch + index;
      ivec2 packedTexShape = ivec2(ceil(float(${n}TexShape[0]) / 2.0), ceil(float(${n}TexShape[1]) / 2.0));
      int texR = index / packedTexShape[1];
      int texC = index - texR * packedTexShape[1];
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(packedTexShape[1], packedTexShape[0]); return ${i.texture2D}(${n}, uv);
    }
  `;let a=e.shapeInfo.logicalShape,o=a.length,s=e.shapeInfo.texShape,c=[Math.ceil(s[0]/2),Math.ceil(s[1]/2)],l=c[0],u=c[1],d=Math.ceil(a[o-1]/2),f=d*Math.ceil(a[o-2]/2),p=`int b, int row, int col`,m=`b * ${f} + (row / 2) * ${d} + (col / 2)`;for(let e=2;e<o-1;e++)p=`int b${e}, `+p,f*=a[o-e-1],m=`b${e} * ${f} + `+m;return`
    vec4 ${r}(${p}) {
      int index = ${m};
      int texR = index / ${u};
      int texC = index - texR * ${u};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${u}, ${l});
      return ${i.texture2D}(${n}, uv);
    }
  `}function OD(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=n[3],o=n[2]*a,s=n[1]*o,{newShape:c,keptDims:l}=w(n);if(c.length<n.length)return`
      ${JE(FD(e,c),t)}
      float ${i}(int row, int col, int depth, int depth2) {
        return ${i}(${ID([`row`,`col`,`depth`,`depth2`],l)});
      }
    `;if(e.shapeInfo.isUniform)return`
      float ${i}(int row, int col, int depth, int depth2) {
        int index = round(dot(vec4(row, col, depth, depth2),
                          vec4(${s}, ${o}, ${a}, 1)));
        ${jD(e)}
      }
    `;let u=e.shapeInfo.flatOffset,d=e.shapeInfo.texShape,f=d[0],p=d[1],m=`int stride2 = ${r}Shape[3];`,h=`int stride1 = ${r}Shape[2] * stride2;`,g=`int stride0 = ${r}Shape[1] * stride1;`;if(p===s&&u==null)return t?`
      float ${i}(int row, int col, int depth, int depth2) {
        ${m}
        ${h}
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(stride1, stride2, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `:`
      float ${i}(int row, int col, int depth, int depth2) {
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(${o}, ${a}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${p}.0, ${f}.0);
        return sampleTexture(${r}, uv);
      }
    `;if(p===a&&u==null)return t?`
      float ${i}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${r}Shape[1] * ${r}Shape[2], ${r}Shape[2], 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `:`
      float ${i}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${n[1]*n[2]}, ${n[2]}, 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${p}.0, ${f}.0);
        return sampleTexture(${r}, uv);
      }
    `;let _=vD(r);return t?`
    float ${i}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      ${m}
      ${h}
      ${g}
      int index = row * stride0 + col * stride1 +
          depth * stride2 + depth2;
      vec2 uv = uvFromFlat(${r}TexShape[0], ${r}TexShape[1], index + ${_});
      return sampleTexture(${r}, uv);
    }
  `:`
    float ${i}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${s} + col * ${o} +
          depth * ${a} + depth2;
      vec2 uv = uvFromFlat(${f}, ${p}, index + ${_});
      return sampleTexture(${r}, uv);
    }
  `}function kD(e){let t=e.shapeInfo.logicalShape,n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1),i=t[4],a=t[3]*i,o=t[2]*a,s=t[1]*o,{newShape:c,keptDims:l}=w(t);if(c.length<t.length)return`
      ${JE(FD(e,c))}
      float ${r}(int row, int col, int depth, int depth2, int depth3) {
        return ${r}(${ID([`row`,`col`,`depth`,`depth2`,`depth3`],l)});
      }
    `;if(e.shapeInfo.isUniform)return`
      float ${r}(int row, int col, int depth, int depth2, int depth3) {
        float index = dot(
          vec4(row, col, depth, depth2),
          vec4(${s}, ${o}, ${a}, ${i})) +
          depth3;
        ${jD(e)}
      }
    `;let u=e.shapeInfo.flatOffset,d=e.shapeInfo.texShape,f=d[0],p=d[1];return p===s&&u==null?`
      float ${r}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
                         vec4(${o}, ${a}, ${i}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${p}.0, ${f}.0);
        return sampleTexture(${n}, uv);
      }
    `:p===i&&u==null?`
      float ${r}(int row, int col, int depth, int depth2, int depth3) {
        float texR = dot(
          vec4(row, col, depth, depth2),
          vec4(${t[1]*t[2]*t[3]},
               ${t[2]*t[3]}, ${t[3]}, 1));
        int texC = depth3;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${p}.0, ${f}.0);
        return sampleTexture(${n}, uv);
      }
    `:`
    float ${r}(int row, int col, int depth, int depth2, int depth3) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${s} + col * ${o} + depth * ${a} +
          depth2 * ${i} + depth3 + ${vD(n)};
      vec2 uv = uvFromFlat(${f}, ${p}, index);
      return sampleTexture(${n}, uv);
    }
  `}function AD(e){let t=e.shapeInfo.logicalShape,n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1),{newShape:i,keptDims:a}=w(t);if(i.length<t.length)return`
      ${JE(FD(e,i))}
      float ${r}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        return ${r}(${ID([`row`,`col`,`depth`,`depth2`,`depth3`,`depth4`],a)});
      }
    `;let o=t[5],s=t[4]*o,c=t[3]*s,l=t[2]*c,u=t[1]*l;if(e.shapeInfo.isUniform)return`
      float ${r}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
        int index = round(dot(
          vec4(row, col, depth, depth2),
          vec4(${u}, ${l}, ${c}, ${s})) +
          dot(
            vec2(depth3, depth4),
            vec2(${o}, 1)));
        ${jD(e)}
      }
    `;let d=e.shapeInfo.flatOffset,f=e.shapeInfo.texShape,p=f[0],m=f[1];return m===u&&d==null?`
      float ${r}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
          vec4(${l}, ${c}, ${s}, ${o})) +
               float(depth4);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${m}.0, ${p}.0);
        return sampleTexture(${n}, uv);
      }
    `:m===o&&d==null?`
      float ${r}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        float texR = dot(vec4(row, col, depth, depth2),
          vec4(${t[1]*t[2]*t[3]*t[4]},
               ${t[2]*t[3]*t[4]},
               ${t[3]*t[4]},
               ${t[4]})) + float(depth3);
        int texC = depth4;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${m}.0, ${p}.0);
        return sampleTexture(${n}, uv);
      }
    `:`
    float ${r}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${u} + col * ${l} + depth * ${c} +
          depth2 * ${s} + depth3 * ${o} + depth4 + ${vD(n)};
      vec2 uv = uvFromFlat(${p}, ${m}, index);
      return sampleTexture(${n}, uv);
    }
  `}function jD(e){let t=e.name,n=g(e.shapeInfo.logicalShape);return n<2?`return ${t};`:`
    for (int i = 0; i < ${n}; i++) {
      if (i == index) {
        return ${t}[i];
      }
    }
  `}function MD(e,t){let n=e.name,r=n.charAt(0).toUpperCase()+n.slice(1),i=`get`+r+`AtOutCoords`,a=e.shapeInfo.logicalShape.length,o=t.logicalShape.length,s=KE(e.shapeInfo.logicalShape,t.logicalShape),c=X(o),l=o-a,u,d=[`x`,`y`,`z`,`w`,`u`,`v`];u=a===0?``:o<2&&s.length>=1?`coords = 0;`:s.map(e=>`coords.${d[e+l]} = 0;`).join(`
`);let f=``;f=o<2&&a>0?`coords`:e.shapeInfo.logicalShape.map((e,t)=>`coords.${d[t+l]}`).join(`, `);let p=`return outputValue;`,m=g(e.shapeInfo.logicalShape)===1,h=g(t.logicalShape)===1;if(a===1&&!m&&!h)p=`
      return vec4(outputValue.xy, outputValue.xy);
    `;else if(m&&!h)p=o===1?`
        return vec4(outputValue.x, outputValue.x, 0., 0.);
      `:`
        return vec4(outputValue.x);
      `;else if(s.length){let e=a-2,t=a-1;s.indexOf(e)>-1&&s.indexOf(t)>-1?p=`return vec4(outputValue.x);`:s.indexOf(e)>-1?p=`return vec4(outputValue.x, outputValue.y, outputValue.x, outputValue.y);`:s.indexOf(t)>-1&&(p=`return vec4(outputValue.xx, outputValue.zz);`)}return`
    vec4 ${i}() {
      ${c} coords = getOutputCoords();
      ${u}
      vec4 outputValue = get${r}(${f});
      ${p}
    }
  `}function ND(e,t){let n=e.name,r=n.charAt(0).toUpperCase()+n.slice(1),i=`get`+r+`AtOutCoords`,a=t.texShape,o=e.shapeInfo.texShape,s=e.shapeInfo.logicalShape.length,c=t.logicalShape.length;if(!e.shapeInfo.isUniform&&s===c&&e.shapeInfo.flatOffset==null&&_(o,a))return`
      float ${i}() {
        return sampleTexture(${n}, resultUV);
      }
    `;let l=X(c),u=KE(e.shapeInfo.logicalShape,t.logicalShape),d=c-s,f,p=[`x`,`y`,`z`,`w`,`u`,`v`];f=s===0?``:c<2&&u.length>=1?`coords = 0;`:u.map(e=>`coords.${p[e+d]} = 0;`).join(`
`);let m=``;return m=c<2&&s>0?`coords`:e.shapeInfo.logicalShape.map((e,t)=>`coords.${p[t+d]}`).join(`, `),`
    float ${i}() {
      ${l} coords = getOutputCoords();
      ${f}
      return get${r}(${m});
    }
  `}function X(e){if(e<=1)return`int`;if(e===2)return`ivec2`;if(e===3)return`ivec3`;if(e===4)return`ivec4`;if(e===5)return`ivec5`;if(e===6)return`ivec6`;throw Error(`GPU for rank ${e} is not yet supported`)}function PD(e,t,n){let{newShape:r,keptDims:i}=w(t),a=t.length,o=e&&a===3&&t[0]===1,s=o?t.slice(1):r,c=!e&&a>1&&!_(t,n)&&r.length<a||o;return{useSqueezeShape:c,uniformShape:c?s:t,keptDims:i}}function FD(e,t){let n=JSON.parse(JSON.stringify(e));return n.shapeInfo.logicalShape=t,n}function ID(e,t){return t.map(t=>e[t]).join(`, `)}function LD(e,t,n,r){let i=n.map((e,n)=>{let r={logicalShape:e.shape,texShape:e.isUniform?null:e.texData.texShape,isUniform:e.isUniform,isPacked:!e.isUniform&&e.texData.isPacked,flatOffset:null};return e.texData!=null&&e.texData.slice!=null&&e.texData.slice.flatOffset>0&&(r.flatOffset=e.texData.slice.flatOffset),{name:t.variableNames[n],shapeInfo:r}}),a=i.map(e=>e.shapeInfo),o={logicalShape:r.shape,texShape:r.texData.texShape,isUniform:!1,isPacked:r.texData.isPacked,flatOffset:null},s=qE(i,o,t),c=ZT(e.gl,s),l=e.createProgram(c);return N().get(`ENGINE_COMPILE_ONLY`)?{program:t,fragmentShader:c,source:s,webGLProgram:l,inShapeInfos:a,outShapeInfo:o,uniformLocations:null,customUniformLocations:null,infLoc:null,nanLoc:null,inShapesLocations:null,inTexShapesLocations:null,outShapeLocation:null,outShapeStridesLocation:null,outTexShapeLocation:null}:Object.assign({program:t,fragmentShader:c,source:s,webGLProgram:l,inShapeInfos:a,outShapeInfo:o},RD(e,t,l))}function RD(e,t,n){let r={},i={},a={},o=[],s,c,l,u=null,d=null;d=e.getUniformLocation(n,`NAN`,!1),N().getNumber(`WEBGL_VERSION`)===1&&(u=e.getUniformLocation(n,`INFINITY`,!1));for(let o=0;o<t.variableNames.length;o++){let s=t.variableNames[o];r[s]=e.getUniformLocation(n,s,!1),r[`offset${s}`]=e.getUniformLocation(n,`offset${s}`,!1),t.enableShapeUniforms&&(i[`${s}Shape`]=e.getUniformLocation(n,`${s}Shape`,!1),a[`${s}TexShape`]=e.getUniformLocation(n,`${s}TexShape`,!1))}return t.enableShapeUniforms&&(s=e.getUniformLocation(n,`outShape`,!1),l=e.getUniformLocation(n,`outShapeStrides`,!1),c=e.getUniformLocation(n,`outTexShape`,!1)),t.customUniforms&&t.customUniforms.forEach((t,r)=>{o[r]=e.getUniformLocation(n,t.name,!1)}),{uniformLocations:r,customUniformLocations:o,infLoc:u,nanLoc:d,inShapesLocations:i,inTexShapesLocations:a,outShapeLocation:s,outShapeStridesLocation:l,outTexShapeLocation:c}}function zD(e,t){if(e.length!==t.length)throw Error(`Binary was compiled with ${e.length} inputs, but was executed with ${t.length} inputs`);e.forEach((e,n)=>{let r=e.logicalShape,i=t[n],a=i.shape;if(!_(r,a))throw Error(`Binary was compiled with different shapes than the current args. Shapes ${r} and ${a} must match`);if(e.isUniform&&i.isUniform)return;let o=e.texShape,s=i.isUniform?null:i.texData.texShape;if(!_(o,s))throw Error(`Binary was compiled with different texture shapes than the current args. Shape ${o} and ${s} must match`)})}function BD(e,t,n,r,i){t.program.enableShapeUniforms||(zD(t.inShapeInfos,n),zD([t.outShapeInfo],[r]));let a=r.texData.texture,o=r.texData.texShape;r.texData.isPacked?e.setOutputPackedMatrixTexture(a.texture,o[0],o[1]):e.setOutputMatrixTexture(a.texture,o[0],o[1]),e.setProgram(t.webGLProgram),N().getNumber(`WEBGL_VERSION`)===1&&t.infLoc!==null&&e.gl.uniform1f(t.infLoc,1/0),t.nanLoc!==null&&e.gl.uniform1f(t.nanLoc,NaN),n.forEach((n,r)=>{let i=t.program.variableNames[r],a=t.uniformLocations[i],o=t.uniformLocations[`offset${i}`],s=t.inShapesLocations[`${i}Shape`],c=t.inTexShapesLocations[`${i}TexShape`];if(s){let{uniformShape:r}=PD(t.program.packedInputs,n.shape,n.texData.texShape);switch(r.length){case 1:e.gl.uniform1iv(s,new Int32Array(r));break;case 2:e.gl.uniform2iv(s,new Int32Array(r));break;case 3:e.gl.uniform3iv(s,new Int32Array(r));break;case 4:e.gl.uniform4iv(s,new Int32Array(r))}}if(c&&e.gl.uniform2i(c,n.texData.texShape[0],n.texData.texShape[1]),a!=null){if(n.isUniform){if(g(n.shape)<2)e.gl.uniform1f(a,n.uniformValues[0]);else{let t=n.uniformValues;t instanceof Float32Array||(t=new Float32Array(t)),e.gl.uniform1fv(a,t)}}else n.texData.slice!=null&&o!=null&&e.gl.uniform1i(o,n.texData.slice.flatOffset),e.setInputMatrixTexture(n.texData.texture.texture,a,r)}});let s=t.outShapeLocation;if(s)switch(r.shape.length){case 1:e.gl.uniform1iv(s,new Int32Array(r.shape));break;case 2:e.gl.uniform2iv(s,new Int32Array(r.shape));break;case 3:e.gl.uniform3iv(s,new Int32Array(r.shape));break;case 4:e.gl.uniform4iv(s,new Int32Array(r.shape))}if(t.outShapeStridesLocation){let n=j(r.shape);switch(r.shape.length){case 2:e.gl.uniform1iv(t.outShapeStridesLocation,new Int32Array(n));break;case 3:e.gl.uniform2iv(t.outShapeStridesLocation,new Int32Array(n));break;case 4:e.gl.uniform3iv(t.outShapeStridesLocation,new Int32Array(n))}}t.outTexShapeLocation&&e.gl.uniform2i(t.outTexShapeLocation,r.texData.texShape[0],r.texData.texShape[1]),t.program.customUniforms&&i&&t.program.customUniforms.forEach((n,r)=>{let a=t.customUniformLocations[r],o=i[r];if(n.type===`float`)e.gl.uniform1fv(a,o);else if(n.type===`vec2`)e.gl.uniform2fv(a,o);else if(n.type===`vec3`)e.gl.uniform3fv(a,o);else if(n.type===`vec4`)e.gl.uniform4fv(a,o);else if(n.type===`int`)e.gl.uniform1iv(a,o);else if(n.type===`ivec2`)e.gl.uniform2iv(a,o);else if(n.type===`ivec3`)e.gl.uniform3iv(a,o);else if(n.type===`ivec4`)e.gl.uniform4iv(a,o);else throw Error(`uniform type ${n.type} is not supported yet.`)}),e.executeProgram()}function VD(e,t,n){let r=``;t.concat(n).forEach(t=>{let i=t.texData!=null&&t.texData.slice!=null&&t.texData.slice.flatOffset>0;if(e.enableShapeUniforms&&!t.isUniform){let a=t.texData.texShape,{useSqueezeShape:o,uniformShape:s,keptDims:c}=PD(e.packedInputs,t.shape,a),l=``,u=``,d=``;if(s.length===1&&e.packedInputs){let e=[Math.ceil(a[0]/2),Math.ceil(a[1]/2)];l=`${e[0]>1}_${e[1]>1}`}else if(s.length===2&&!e.packedInputs)u=`${s[0]>1}_${s[1]>1}`;else if(s.length>2&&!e.packedInputs){let e=j(s);d=`${e[0]===a[1]}_${e[e.length-1]===a[1]}`}let f=t.shape.length,p=s.length===2&&_(t.shape,a),m=g(t.shape)===1,h=rs(t.shape,n.shape),v=!e.packedInputs&&f===n.shape.length&&_(a,n.texData.texShape),y=e.packedInputs||s.length>2?``:`${a[0]>1}_${a[1]>1}`;r+=`${f}_${v}_${o?c:``}_${s.length}_${m}_${h}_${p}_${l}_${u}_${d}_${y}_${i}`}else{let e=t.isUniform?`uniform`:t.texData.texShape;r+=`${t.shape}_${e}_${i}`}});let i=e.userCode,a=e.constructor.name;return a+=`_`+r+`_`+i+`${N().getNumber(`WEBGL_VERSION`)}`,a}function HD(e){return N().getBool(`WEBGL_USE_SHAPES_UNIFORMS`)&&e<=4}var UD=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!1,this.packedOutput=!0,this.outPackingScheme=FT.DENSE,this.customUniforms=[{name:`texShape`,type:`ivec2`}];let t=RE();this.outputShape=e,this.enableShapeUniforms=HD(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?BE([`r`,`c`,`d`],e):zE([`r`,`c`,`d`],e)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getA(rc.x, rc.y, rc.z);
        }

        ${t.output} = result;
      }
    `}},WD=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.outPackingScheme=FT.DENSE,this.customUniforms=[{name:`texShape`,type:`ivec2`}];let t=RE();this.outputShape=e,this.enableShapeUniforms=HD(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?BE([`r`,`c`,`d`],e):zE([`r`,`c`,`d`],e)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getChannel(getA(rc.x, rc.y, rc.z), vec2(rc.y, rc.z));
        }

        ${t.output} = result;
      }
    `}},GD=class{constructor(e){this.variableNames=[`A`],this.outTexUsage=IT.DOWNLOAD;let t=RE();this.outputShape=e,this.userCode=`
      ${GE}

      void main() {
        float x = getAAtOutCoords();
        ${t.output} = encode_float(x);
      }
    `}},KD=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!1,this.outTexUsage=IT.DOWNLOAD;let t=RE();this.outputShape=e,this.userCode=`
      ${GE}

      void main() {
        ivec3 coords = getOutputCoords();
        float x = getChannel(getAAtOutCoords(), vec2(coords.y, coords.z));
        ${t.output} = encode_float(x);
      }
    `}},qD=class{constructor(e,t=!1){this.variableNames=[`A`],this.customUniforms=[{name:`texShape`,type:`ivec2`}];let n=RE();this.outputShape=e,this.enableShapeUniforms=HD(this.outputShape.length);let r=`result`;t&&(r=`floor(result * 255. + 0.5)`),this.userCode=`
      ${this.enableShapeUniforms?WE():UE(e)}

      void main() {
        ivec3 coords = getOutputCoords();

        int flatIndex = getFlatIndex(coords);
        int offset = imod(flatIndex, 4);

        flatIndex = idiv(flatIndex, 4, 1.);

        int r = flatIndex / texShape[1];
        int c = imod(flatIndex, texShape[1]);
        vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
        vec4 values = ${n.texture2D}(A, uv);

        float result;

        if(offset == 0) {
          result = values[0];
        } else if(offset == 1) {
          result = values[1];
        } else if(offset == 2) {
          result = values[2];
        } else {
          result = values[3];
        }

        ${n.output} = vec4(${r}, 0., 0., 0.);
      }
    `}},JD=class{constructor(e,t=!1){this.variableNames=[`A`],this.packedInputs=!1,this.packedOutput=!0,this.customUniforms=[{name:`texShape`,type:`ivec2`}];let n=RE();this.outputShape=e,this.enableShapeUniforms=HD(this.outputShape.length);let r=``,i=`result`;t&&(i=`floor(result * 255. + 0.5)`);for(let t=0;t<=1;t++)for(let i=0;i<=1;i++){let a=t*2+i;r+=`
          localCoords = coords;
          if(localCoords[2] + ${i} < ${this.enableShapeUniforms?`outShape[2]`:`${e[2]}`}) {
          localCoords[2] += ${i};
          if (localCoords[1] + ${t} < ${this.enableShapeUniforms?`outShape[1]`:`${e[1]}`}) {
            localCoords[1] += ${t};

            flatIndex = getFlatIndex(localCoords);
            offset = imod(flatIndex, 4);

            flatIndex = idiv(flatIndex, 4, 1.);

            int r = flatIndex / texShape[1];
            int c = imod(flatIndex, texShape[1]);
            vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
            values = ${n.texture2D}(A, uv);

            if (offset == 0) {
              result[${a}] = values[0];
            } else if (offset == 1) {
              result[${a}] = values[1];
            } else if (offset == 2) {
              result[${a}] = values[2];
            } else {
              result[${a}] = values[3];
            }
          }
        }
        `}this.userCode=`
        ${this.enableShapeUniforms?WE():UE(e)}

        void main() {
          ivec3 coords = getOutputCoords();

          vec4 result = vec4(0.);
          int flatIndex, r, c, offset;
          ivec3 localCoords;
          vec2 uv;
          vec4 values;

          ${r}

          ${n.output} = ${i};
        }
    `}};function YD(e){let t=RE();return XT(e,`${t.version}
    precision highp float;
    ${t.attribute} vec3 clipSpacePos;
    ${t.attribute} vec2 uv;
    ${t.varyingVs} vec2 resultUV;

    void main() {
      gl_Position = vec4(clipSpacePos, 1);
      resultUV = uv;
    }`)}function XD(e){return rE(e,new Float32Array([-1,1,0,0,1,-1,-1,0,0,0,1,1,0,1,1,1,-1,0,1,0]))}function ZD(e){return iE(e,new Uint16Array([0,1,2,2,1,3]))}function QD(e,t,n,r,i,a){oE(t,n);let o=aE(e),s=e.TEXTURE_2D;return J(e,()=>e.bindTexture(s,o)),J(e,()=>e.texParameteri(s,e.TEXTURE_WRAP_S,e.CLAMP_TO_EDGE)),J(e,()=>e.texParameteri(s,e.TEXTURE_WRAP_T,e.CLAMP_TO_EDGE)),J(e,()=>e.texParameteri(s,e.TEXTURE_MIN_FILTER,e.NEAREST)),J(e,()=>e.texParameteri(s,e.TEXTURE_MAG_FILTER,e.NEAREST)),N().getNumber(`WEBGL_VERSION`)===1?J(e,()=>e.texImage2D(s,0,r,t,n,0,i,a,null)):J(e,()=>e.texStorage2D(s,1,r,t,n)),J(e,()=>e.bindTexture(e.TEXTURE_2D,null)),{texture:o,texShape:[n,t]}}function $D(e){return e.internalFormatFloat}function eO(e,t,n,r){let[i,a]=RT(t,n);return QD(e,i,a,$D(r),r.textureFormatFloat,e.FLOAT)}function tO(e){return e.internalFormatHalfFloat}function nO(e,t,n,r){let[i,a]=RT(t,n);return QD(e,i,a,tO(r),r.textureFormatFloat,r.textureTypeHalfFloat)}function rO(e){return e.downloadTextureFormat}function iO(e,t,n,r){let[i,a]=RT(t,n);return QD(e,i,a,rO(r),e.RGBA,e.UNSIGNED_BYTE)}function aO(e){return e.internalFormatPackedFloat}function oO(e,t,n,r){let[i,a]=VT(t,n);return QD(e,i,a,aO(r),e.RGBA,e.FLOAT)}function sO(e){return e.internalFormatPackedHalfFloat}function cO(e,t,n,r){let[i,a]=VT(t,n);return QD(e,i,a,sO(r),e.RGBA,r.textureTypeHalfFloat)}function lO(e,t,n){return J(e,()=>e.bindBuffer(e.ARRAY_BUFFER,n)),cE(e,t,`clipSpacePos`,n,3,20,0)&&cE(e,t,`uv`,n,2,20,12)}function uO(e,t,n,r,i,a){J(e,()=>e.bindTexture(e.TEXTURE_2D,t));let o,s,c;i instanceof Uint8Array?(o=new Uint8Array(n*r*4),s=e.UNSIGNED_BYTE,c=e.RGBA):(o=new Float32Array(n*r*4),s=e.FLOAT,c=a.internalFormatPackedFloat),o.set(i),N().getNumber(`WEBGL_VERSION`)===2?J(e,()=>e.texSubImage2D(e.TEXTURE_2D,0,0,0,n,r,e.RGBA,s,o)):J(e,()=>e.texImage2D(e.TEXTURE_2D,0,c,n,r,0,e.RGBA,s,o)),J(e,()=>e.bindTexture(e.TEXTURE_2D,null))}function dO(e,t,n){J(e,()=>e.bindTexture(e.TEXTURE_2D,t)),n.data instanceof Uint8Array?N().getNumber(`WEBGL_VERSION`)===2?J(e,()=>e.texSubImage2D(e.TEXTURE_2D,0,0,0,n.width,n.height,e.RGBA,e.UNSIGNED_BYTE,n.data)):J(e,()=>e.texImage2D(e.TEXTURE_2D,0,e.RGBA,n.width,n.height,0,e.RGBA,e.UNSIGNED_BYTE,n.data)):N().getNumber(`WEBGL_VERSION`)===2?J(e,()=>e.texSubImage2D(e.TEXTURE_2D,0,0,0,e.RGBA,e.UNSIGNED_BYTE,n)):J(e,()=>e.texImage2D(e.TEXTURE_2D,0,e.RGBA,e.RGBA,e.UNSIGNED_BYTE,n)),J(e,()=>e.bindTexture(e.TEXTURE_2D,null))}function fO(e,t,n,r){let i=e.createBuffer();J(e,()=>e.bindBuffer(e.PIXEL_PACK_BUFFER,i));let a=16*t*n;return J(e,()=>e.bufferData(e.PIXEL_PACK_BUFFER,a,e.STREAM_READ)),J(e,()=>e.readPixels(0,0,n,t,e.RGBA,e.FLOAT,0)),J(e,()=>e.bindBuffer(e.PIXEL_PACK_BUFFER,null)),i}function pO(e,t,n){let r=e,i=new Float32Array(n);return r.bindBuffer(r.PIXEL_PACK_BUFFER,t),r.getBufferSubData(r.PIXEL_PACK_BUFFER,0,i),r.bindBuffer(r.PIXEL_PACK_BUFFER,null),i}function mO(e,t,n,r){let[i,a]=RT(t,n),o=new Uint8Array(zT(t*n,4));return J(e,()=>e.readPixels(0,0,i,a,r.downloadTextureFormat,e.UNSIGNED_BYTE,o)),new Float32Array(o.buffer)}function hO(e,t,n,r,i,a,o,s){let c=e,l=new Float32Array(HT(a,o));return c.bindBuffer(c.PIXEL_PACK_BUFFER,t),c.getBufferSubData(c.PIXEL_PACK_BUFFER,0,l),c.bindBuffer(c.PIXEL_PACK_BUFFER,null),l}function gO(e,t,n){let r=new Float32Array(t*n*4);return J(e,()=>e.readPixels(0,0,n,t,e.RGBA,e.FLOAT,r)),r}var _O=class{constructor(e){this.outputTexture=null,this.program=null,this.disposed=!1,this.vertexAttrsAreBound=!1,this.itemsToPoll=[];let t=N().getNumber(`WEBGL_VERSION`);e==null?this.gl=MT(t):(this.gl=e,jT(t,e));let n=`WEBGL_color_buffer_float`,r=`EXT_color_buffer_half_float`;if(this.parallelCompilationExtension=this.gl.getExtension(`KHR_parallel_shader_compile`),N().getNumber(`WEBGL_VERSION`)===1){let e=`OES_texture_half_float`;if(this.textureFloatExtension=YT(this.gl,`OES_texture_float`),AE(this.gl,e))this.textureHalfFloatExtension=YT(this.gl,e);else if(N().get(`WEBGL_FORCE_F16_TEXTURES`))throw Error(`GL context does not support half float textures, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.`);if(this.colorBufferFloatExtension=this.gl.getExtension(n),AE(this.gl,r))this.colorBufferHalfFloatExtension=YT(this.gl,r);else if(N().get(`WEBGL_FORCE_F16_TEXTURES`))throw Error(`GL context does not support color renderable half floats, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.`)}else if(n=`EXT_color_buffer_float`,AE(this.gl,n))this.colorBufferFloatExtension=this.gl.getExtension(n);else if(AE(this.gl,r))this.colorBufferHalfFloatExtension=this.gl.getExtension(r);else throw Error(`GL context does not support color renderable floats`);this.vertexBuffer=XD(this.gl),this.indexBuffer=ZD(this.gl),this.framebuffer=sE(this.gl),this.textureConfig=UT(this.gl,this.textureHalfFloatExtension)}get debug(){return N().getBool(`DEBUG`)}dispose(){if(this.disposed)return;this.program!=null&&console.warn(`Disposing a GPGPUContext that still has a bound WebGLProgram. This is probably a resource leak, delete the program with GPGPUContext.deleteProgram before disposing.`),this.outputTexture!=null&&console.warn(`Disposing a GPGPUContext that still has a bound output matrix texture.  This is probably a resource leak, delete the output matrix texture with GPGPUContext.deleteMatrixTexture before disposing.`);let e=this.gl;J(e,()=>e.finish()),J(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,null)),J(e,()=>e.deleteFramebuffer(this.framebuffer)),J(e,()=>e.bindBuffer(e.ARRAY_BUFFER,null)),J(e,()=>e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,null)),J(e,()=>e.deleteBuffer(this.indexBuffer)),this.disposed=!0}createFloat32MatrixTexture(e,t){return this.throwIfDisposed(),eO(this.gl,e,t,this.textureConfig)}createFloat16MatrixTexture(e,t){return this.throwIfDisposed(),nO(this.gl,e,t,this.textureConfig)}createUnsignedBytesMatrixTexture(e,t){return this.throwIfDisposed(),iO(this.gl,e,t,this.textureConfig)}uploadPixelDataToTexture(e,t){this.throwIfDisposed(),dO(this.gl,e,t)}uploadDenseMatrixToTexture(e,t,n,r){this.throwIfDisposed(),uO(this.gl,e,t,n,r,this.textureConfig)}createFloat16PackedMatrixTexture(e,t){return this.throwIfDisposed(),cO(this.gl,e,t,this.textureConfig)}createPackedMatrixTexture(e,t){return this.throwIfDisposed(),oO(this.gl,e,t,this.textureConfig)}deleteMatrixTexture(e){this.throwIfDisposed(),this.outputTexture===e&&(mE(this.gl,this.framebuffer),this.outputTexture=null),J(this.gl,()=>this.gl.deleteTexture(e))}downloadByteEncodedFloatMatrixFromOutputTexture(e,t,n){return this.downloadMatrixDriver(e,()=>mO(this.gl,t,n,this.textureConfig))}downloadPackedMatrixFromBuffer(e,t,n,r,i,a){return hO(this.gl,e,t,n,r,i,a,this.textureConfig)}downloadFloat32MatrixFromBuffer(e,t){return pO(this.gl,e,t)}createBufferFromTexture(e,t,n){this.bindTextureToFrameBuffer(e);let r=fO(this.gl,t,n,this.textureConfig);return this.unbindTextureToFrameBuffer(),r}createAndWaitForFence(){let e=this.createFence(this.gl);return this.pollFence(e)}createFence(e){let t,n;if(N().getBool(`WEBGL_FENCE_API_ENABLED`)){let r=e,i=r.fenceSync(r.SYNC_GPU_COMMANDS_COMPLETE,0);e.flush(),n=()=>{let e=r.clientWaitSync(i,0,0);return e===r.ALREADY_SIGNALED||e===r.CONDITION_SATISFIED},t=i}else N().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)>0?(t=this.beginQuery(),this.endQuery(),n=()=>this.isQueryAvailable(t,N().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`))):n=()=>!0;return{query:t,isFencePassed:n}}downloadMatrixFromPackedTexture(e,t,n){return this.downloadMatrixDriver(e,()=>gO(this.gl,t,n))}createProgram(e){this.throwIfDisposed();let t=this.gl;this.vertexShader??=YD(t);let n=eE(t);return J(t,()=>t.attachShader(n,this.vertexShader)),J(t,()=>t.attachShader(n,e)),tE(t,n),this.debug&&nE(t,n),this.vertexAttrsAreBound||=(this.setProgram(n),lO(t,this.program,this.vertexBuffer)),n}deleteProgram(e){this.throwIfDisposed(),e===this.program&&(this.program=null),e!=null&&J(this.gl,()=>this.gl.deleteProgram(e))}setProgram(e){this.throwIfDisposed(),this.program=e,this.program!=null&&this.debug&&nE(this.gl,this.program),J(this.gl,()=>this.gl.useProgram(e))}getUniformLocation(e,t,n=!0){return this.throwIfDisposed(),n?uE(this.gl,e,t):dE(this.gl,e,t)}getAttributeLocation(e,t){return this.throwIfDisposed(),J(this.gl,()=>this.gl.getAttribLocation(e,t))}getUniformLocationNoThrow(e,t){return this.throwIfDisposed(),this.gl.getUniformLocation(e,t)}setInputMatrixTexture(e,t,n){this.throwIfDisposed(),this.throwIfNoProgram(),fE(this.gl,e,t,n)}setOutputMatrixTexture(e,t,n){this.setOutputMatrixTextureDriver(e,n,t)}setOutputPackedMatrixTexture(e,t,n){this.throwIfDisposed();let[r,i]=VT(t,n);this.setOutputMatrixTextureDriver(e,r,i)}setOutputMatrixWriteRegion(e,t,n,r){this.setOutputMatrixWriteRegionDriver(n,e,r,t)}setOutputPackedMatrixWriteRegion(e,t,n,r){throw Error(`setOutputPackedMatrixWriteRegion not implemented.`)}debugValidate(){this.program!=null&&nE(this.gl,this.program),hE(this.gl)}executeProgram(){this.throwIfDisposed(),this.throwIfNoProgram();let e=this.gl;this.debug&&this.debugValidate(),J(e,()=>e.drawElements(e.TRIANGLES,6,e.UNSIGNED_SHORT,0))}blockUntilAllProgramsCompleted(){this.throwIfDisposed(),J(this.gl,()=>this.gl.finish())}getQueryTimerExtension(){return this.disjointQueryTimerExtension??=YT(this.gl,N().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)===2?`EXT_disjoint_timer_query_webgl2`:`EXT_disjoint_timer_query`),this.disjointQueryTimerExtension}getQueryTimerExtensionWebGL2(){return this.getQueryTimerExtension()}getQueryTimerExtensionWebGL1(){return this.getQueryTimerExtension()}beginQuery(){if(N().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)===2){let e=this.gl,t=this.getQueryTimerExtensionWebGL2(),n=e.createQuery();return e.beginQuery(t.TIME_ELAPSED_EXT,n),n}let e=this.getQueryTimerExtensionWebGL1(),t=e.createQueryEXT();return e.beginQueryEXT(e.TIME_ELAPSED_EXT,t),t}endQuery(){if(N().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)===2){let e=this.gl,t=this.getQueryTimerExtensionWebGL2();e.endQuery(t.TIME_ELAPSED_EXT);return}let e=this.getQueryTimerExtensionWebGL1();e.endQueryEXT(e.TIME_ELAPSED_EXT)}async waitForQueryAndGetTime(e){return await x(()=>this.disposed||this.isQueryAvailable(e,N().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`))),this.getQueryTime(e,N().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`))}getQueryTime(e,t){if(t===0)return null;if(t===2){let t=this.gl;return t.getQueryParameter(e,t.QUERY_RESULT)/1e6}{let t=this.getQueryTimerExtensionWebGL1();return t.getQueryObjectEXT(e,t.QUERY_RESULT_EXT)/1e6}}isQueryAvailable(e,t){if(t===0)return!0;if(t===2){let t=this.gl,n=this.getQueryTimerExtensionWebGL2(),r=t.getQueryParameter(e,t.QUERY_RESULT_AVAILABLE);return this.disjoint??=this.gl.getParameter(n.GPU_DISJOINT_EXT),r&&!this.disjoint}{let t=this.getQueryTimerExtensionWebGL1(),n=t.getQueryObjectEXT(e,t.QUERY_RESULT_AVAILABLE_EXT);return this.disjoint??=this.gl.getParameter(t.GPU_DISJOINT_EXT),n&&!this.disjoint}}pollFence(e){return new Promise(t=>{this.addItemToPoll(()=>e.isFencePassed(),()=>t())})}pollItems(){let e=vO(this.itemsToPoll.map(e=>e.isDoneFn));for(let t=0;t<=e;++t){let{resolveFn:e}=this.itemsToPoll[t];e()}this.itemsToPoll=this.itemsToPoll.slice(e+1)}addItemToPoll(e,t){if(this.itemsToPoll.push({isDoneFn:e,resolveFn:t}),this.itemsToPoll.length>1)return;let n;`setTimeoutCustom`in N().platform&&(n=N().platform.setTimeoutCustom.bind(N().platform)),x(()=>(this.pollItems(),this.itemsToPoll.length===0),()=>0,null,n)}bindTextureToFrameBuffer(e){this.throwIfDisposed(),pE(this.gl,e,this.framebuffer),this.debug&&hE(this.gl)}unbindTextureToFrameBuffer(){this.outputTexture==null?mE(this.gl,this.framebuffer):(pE(this.gl,this.outputTexture,this.framebuffer),this.debug&&hE(this.gl))}downloadMatrixDriver(e,t){this.bindTextureToFrameBuffer(e);let n=t();return this.unbindTextureToFrameBuffer(),n}setOutputMatrixTextureDriver(e,t,n){this.throwIfDisposed();let r=this.gl;pE(r,e,this.framebuffer),this.debug&&hE(r),this.outputTexture=e,J(r,()=>r.viewport(0,0,t,n)),J(r,()=>r.scissor(0,0,t,n))}setOutputMatrixWriteRegionDriver(e,t,n,r){this.throwIfDisposed(),J(this.gl,()=>this.gl.scissor(e,t,n,r))}throwIfDisposed(){if(this.disposed)throw Error(`Attempted to use disposed GPGPUContext.`)}throwIfNoProgram(){if(this.program==null)throw Error(`No GPU program is currently set.`)}};function vO(e){let t=0;for(;t<e.length&&e[t]();++t);return t-1}var{addImpl:yO,bincountImpl:bO,bincountReduceImpl:xO,castImpl:SO,ceilImpl:CO,concatImpl:wO,equalImpl:TO,expImpl:EO,expm1Impl:DO,floorImpl:OO,gatherNdImpl:kO,gatherV2Impl:AO,greaterImpl:jO,greaterEqualImpl:MO,lessImpl:NO,lessEqualImpl:PO,linSpaceImpl:FO,logImpl:IO,maxImpl:LO,maximumImpl:RO,minimumImpl:zO,multiplyImpl:BO,negImpl:VO,notEqualImpl:HO,prodImpl:UO,raggedGatherImpl:WO,raggedTensorToTensorImpl:GO,rangeImpl:KO,rsqrtImpl:qO,scatterImpl:JO,sigmoidImpl:YO,simpleAbsImpl:XO,sliceImpl:ZO,sparseFillEmptyRowsImpl:QO,sparseReshapeImpl:$O,sparseSegmentReductionImpl:ek,sqrtImpl:tk,stridedSliceImpl:nk,stringNGramsImpl:rk,stringSplitImpl:ik,stringToHashBucketFastImpl:ak,subImpl:ok,tileImpl:sk,topKImpl:ck,transposeImpl:lk,uniqueImpl:uk}=Eb;function dk(e,t){return[`x`,`y`,`z`,`w`,`u`,`v`].slice(0,t).map(t=>`${e}.${t}`)}function fk(e,t){return t===1?[e]:dk(e,t)}function pk(e,t){if(e===1)return`rc`;let n=``;for(let r=0;r<e;r++)n+=t[r],r<e-1&&(n+=`,`);return n}var mk=class{constructor(e){if(this.variableNames=[`A`],this.packedInputs=!1,this.packedOutput=!0,this.outputShape=e,this.rank=e.length,this.enableShapeUniforms=HD(this.outputShape.length),this.rank===0)this.userCode=`
        void main() {
          setOutput(vec4(getA(), 0., 0., 0.));
        }
      `;else{let e=fk(`rc`,this.rank),t=X(this.rank),n=this.getOutOfBoundsCondition(e),r=this.getSetup(e),i=this.getOutput(e);this.userCode=`
        void main() {
          ${t} rc = getOutputCoords();

          if(${n}) {
            setOutput(vec4(0));
          } else {
            ${r}

            setOutput(vec4(${i}));
          }
        }
      `}}getSourceCoordsArr(e){let t=[];for(let n=0;n<=1;n++)for(let r=0;r<=1;r++){let i=`${n===0?`r`:`rp1`}, ${r===0?`c`:`cp1`}`;for(let t=2;t<this.rank;t++)i=`${e[e.length-1-t]},`+i;t.push(i)}return t}getOutOfBoundsCondition(e){if(this.rank===1)return`rc > ${this.enableShapeUniforms?`outShape`:this.outputShape[0]}`;let t=``;for(let n=this.rank-2;n<this.rank;n++)t+=`${e[n]} >= ${this.enableShapeUniforms?`outShape[${n}]`:this.outputShape[n]}`,n<this.rank-1&&(t+=`||`);return t}getSetup(e){if(this.rank===1)return``;let t=e.slice(-2),n=this.enableShapeUniforms?`outShape[${this.rank} - 1]`:this.outputShape[this.rank-1],r=this.enableShapeUniforms?`outShape[${this.rank} - 2]`:this.outputShape[this.rank-2];return`
      int r = ${t[0]};
      int c = ${t[1]};
      int rp1 = r + 1;
      int cp1 = c + 1;

      bool cEdge = cp1 >= ${n};
      bool rEdge = rp1 >= ${r};
    `}getOutput(e){let t=this.getSourceCoordsArr(e);return this.rank===1?`getA(rc), (rc + 1 >= ${this.enableShapeUniforms?`outShape`:this.outputShape[0]} ? 0. : getA(rc + 1)), 0, 0`:`getA(${t[0]}),
            cEdge ? 0. : getA(${t[1]}),
            rEdge ? 0. : getA(${t[2]}),
            rEdge || cEdge ? 0. : getA(${t[3]})`}},hk=class{constructor(e,t){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`inputShape`,type:`ivec3`}],this.outputShape=e,this.enableShapeUniforms=HD(this.outputShape.length);let n=``;for(let e=0;e<4;e++){let t=`thisRC = rc;`;e%2==1&&(t+=`thisRC.z += 1;`),e>1&&(t+=`thisRC.y += 1;`),n+=`
        ${t}
        ${e>0?`if(thisRC.y < rows && thisRC.z < cols){`:``}
          int flatIndex = getFlatIndex(thisRC);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

          result[${e}] =
            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
        ${e>0?`}`:``}
      `}this.userCode=`
      ${gk(t,this.enableShapeUniforms)}
      ${this.enableShapeUniforms?WE():UE(e)}

      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.);

        ivec3 thisRC;
        int rows = ${this.enableShapeUniforms?`outShape[1]`:e[1]};
        int cols = ${this.enableShapeUniforms?`outShape[2]`:e[2]};

        ${n}

        setOutput(result);
      }
    `}};function gk(e,t){return`
    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      ${t?HE([`r`,`c`,`d`],`inputShape`):zE([`r`,`c`,`d`],e)}
      return ivec3(r, c, d);
    }
  `}var _k=class{constructor(e){this.gpgpu=e,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0,this.freeTextures={},this.logEnabled=!1,this.usedTextures={}}acquireTexture(e,t,n){let r=Sk(t,n),i=Ck(e,r,n);i in this.freeTextures||(this.freeTextures[i]=[]),i in this.usedTextures||(this.usedTextures[i]=[]);let a=yk(e,r,this.gpgpu.gl,this.gpgpu.textureConfig,n);if(this.freeTextures[i].length>0){this.numFreeTextures--,this.numUsedTextures++,this._numBytesFree-=a,this.log();let e=this.freeTextures[i].shift();return this.usedTextures[i].push(e),e}let o;return r===LT.PACKED_2X2_FLOAT32?o=this.gpgpu.createPackedMatrixTexture(e[0],e[1]):r===LT.PACKED_2X2_FLOAT16?o=this.gpgpu.createFloat16PackedMatrixTexture(e[0],e[1]):r===LT.UNPACKED_FLOAT32?o=this.gpgpu.createFloat32MatrixTexture(e[0],e[1]):r===LT.UNPACKED_FLOAT16?o=this.gpgpu.createFloat16MatrixTexture(e[0],e[1]):r===LT.PACKED_4X1_UNSIGNED_BYTE&&(o=this.gpgpu.createUnsignedBytesMatrixTexture(e[0],e[1])),this.usedTextures[i].push(o),this.numUsedTextures++,this._numBytesAllocated+=a,this.log(),o}releaseTexture(e,t,n,r){if(this.freeTextures==null)return;let i=Sk(n,r),a=Ck(t,i,r);a in this.freeTextures||(this.freeTextures[a]=[]);let o=yk(t,i,this.gpgpu.gl,this.gpgpu.textureConfig,r),s=N().get(`WEBGL_DELETE_TEXTURE_THRESHOLD`);s!==-1&&this._numBytesAllocated>s?(this.gpgpu.deleteMatrixTexture(e.texture),this._numBytesAllocated-=o):(this.freeTextures[a].push(e),this.numFreeTextures++,this._numBytesFree+=o),this.numUsedTextures--;let c=this.usedTextures[a],l=c.indexOf(e);if(l<0)throw Error(`Cannot release a texture that was never provided by this texture manager`);c.splice(l,1),this.log()}log(){if(!this.logEnabled)return;let e=this.numFreeTextures+this.numUsedTextures;console.log(`Free/Used`,`${this.numFreeTextures} / ${this.numUsedTextures}`,`(${e})`);let t=this._numBytesFree/this._numBytesAllocated;console.log(`Bytes allocated: ${this._numBytesAllocated}`),console.log(`Bytes unused: ${this._numBytesFree} (${Math.round(100*t)}%)`)}get numBytesAllocated(){return this._numBytesAllocated}get numBytesFree(){return this._numBytesFree}getNumUsedTextures(){return this.numUsedTextures}getNumFreeTextures(){return this.numFreeTextures}dispose(){if(this.freeTextures!=null){for(let e in this.freeTextures)this.freeTextures[e].forEach(e=>{this.gpgpu.deleteMatrixTexture(e.texture)});for(let e in this.usedTextures)this.usedTextures[e].forEach(e=>{this.gpgpu.deleteMatrixTexture(e.texture)});this.freeTextures=null,this.usedTextures=null,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0}}};function vk(e,t){let n=e;if(t===n.R32F)return 4;if(t===n.R16F)return 2;if(t===n.RGBA32F||t===e.RGBA)return 16;if(t===n.RGBA16F)return 8;if(t===n.RGBA8)return 4;throw Error(`Unknown internal format ${t}`)}function yk(e,t,n,r,i){let a=bk(t,r),o;if(i){let[t,n]=VT(e[0],e[1]);o=t*n}else{let[t,n]=RT(e[0],e[1]);o=t*n}let s=vk(n,a);return o*s}function bk(e,t){switch(e){case LT.PACKED_2X2_FLOAT32:return aO(t);case LT.PACKED_2X2_FLOAT16:return sO(t);case LT.UNPACKED_FLOAT32:return $D(t);case LT.UNPACKED_FLOAT16:return tO(t);case LT.PACKED_4X1_UNSIGNED_BYTE:return rO(t);default:throw Error(`Unknown physical texture type ${e}`)}}function xk(e){return N().getBool(`WEBGL_RENDER_FLOAT32_ENABLED`)?e?LT.PACKED_2X2_FLOAT32:LT.UNPACKED_FLOAT32:e?LT.PACKED_2X2_FLOAT16:LT.UNPACKED_FLOAT16}function Sk(e,t){if(e===IT.UPLOAD)return LT.PACKED_2X2_FLOAT32;if(e===IT.RENDER||e==null)return xk(t);if(e===IT.DOWNLOAD||e===IT.PIXELS)return LT.PACKED_4X1_UNSIGNED_BYTE;throw Error(`Unknown logical texture type ${e}`)}function Ck(e,t,n){return`${e[0]}_${e[1]}_${t}_${n}`}var wk=class{constructor(e,t){this.variableNames=[`A`],this.outputShape=e,this.enableShapeUniforms=HD(this.outputShape.length),this.userCode=`
      float unaryOperation(float x) {
        ${t}
      }

      void main() {
        float x = getAAtOutCoords();
        float y = unaryOperation(x);

        setOutput(y);
      }
    `}},Tk=`if (isnan(x)) return x;`,Ek=`return x;`,Dk=`return abs(x);`,Ok=`return (x >= 0.0) ? x : (exp(x) - 1.0);`,kk=Tk+`
  return (x < 0.0) ? 0.0 : x;
`,Ak=Tk+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,jk=`return x;`,Mk=`return 1.0 / (1.0 + exp(-1.0 * x));`,Nk=`return x;`,Pk=`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`,Fk=`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,Ik=`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,Lk=`return 1.0 / (1.0 + exp(-1.0 * x));`,Rk=class{constructor(e,t){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.enableShapeUniforms=HD(this.outputShape.length),this.userCode=`
      vec4 unaryOperation(vec4 x) {
        ${t}
      }

      void main() {
        vec4 x = getAAtOutCoords();
        vec4 y = unaryOperation(x);

        setOutput(y);
      }
    `}},zk=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!1,this.outputShape=e,this.enableShapeUniforms=HD(this.outputShape.length);let t=e.length,n=fk(`rc`,t),r=X(t),i=pk(t,n),a=n.slice(-2),o=t<=1?`rc`:`vec2(${a.join(`,`)})`;this.userCode=`
      void main() {
        ${r} rc = getOutputCoords();
        vec4 packedInput = getA(${i});

        setOutput(getChannel(packedInput, ${o}));
      }
    `}},Bk=zm,Vk=1e-7,Hk=1e-4,Uk={};function Wk(e){return e in Uk||(Uk[e]={}),Uk[e]}var Gk=N().getNumber(`CPU_HANDOFF_SIZE_THRESHOLD`),Kk=600;function qk(){return N().global.screen==null?1024:N().global.screen.height*N().global.screen.width*window.devicePixelRatio*Kk/1024/1024}var Jk=class e extends o{constructor(e){if(super(),this.pendingRead=new WeakMap,this.pendingDisposal=new WeakSet,this.dataRefCount=new WeakMap,this.numBytesInGPU=0,this.uploadWaitMs=0,this.downloadWaitMs=0,this.lastGlFlushTime=0,this.warnedAboutMemory=!1,this.pendingDeletes=0,this.disposed=!1,!N().getBool(`HAS_WEBGL`))throw Error(`WebGL is not supported on this device`);let t;e==null?(t=new _O(MT(N().getNumber(`WEBGL_VERSION`))),this.binaryCache=Wk(N().getNumber(`WEBGL_VERSION`)),this.gpgpuCreatedLocally=!0):(t=e instanceof _O?e:new _O(MT(N().getNumber(`WEBGL_VERSION`),e)),this.binaryCache={},this.gpgpuCreatedLocally=!1),this.gpgpu=t,this.canvas=this.gpgpu.gl.canvas,this.textureManager=new _k(this.gpgpu),this.numMBBeforeWarning=qk(),this.texData=new a(this,Bo())}nextDataId(){return e.nextDataId++}numDataIds(){return this.texData.numDataIds()-this.pendingDeletes}write(e,t,n){if((N().getBool(`WEBGL_CHECK_NUMERICAL_PROBLEMS`)||N().getBool(`DEBUG`))&&this.checkNumericalProblems(e),n===`complex64`&&e!=null)throw Error(`Cannot write to a complex64 dtype. Please use tf.complex(real, imag).`);let r={id:this.nextDataId()};return this.texData.set(r,{shape:t,dtype:n,values:e,usage:IT.UPLOAD,refCount:1}),r}refCount(e){return this.texData.has(e)?this.texData.get(e).refCount:0}incRef(e){let t=this.texData.get(e);t.refCount++}decRef(e){if(this.texData.has(e)){let t=this.texData.get(e);t.refCount--}}move(e,t,n,r,i){if(N().getBool(`DEBUG`)&&this.checkNumericalProblems(t),r===`complex64`)throw Error(`Cannot write to a complex64 dtype. Please use tf.complex(real, imag).`);this.texData.set(e,{shape:n,dtype:r,values:t,usage:IT.UPLOAD,refCount:i})}disposeIntermediateTensorInfo(e){this.disposeData(e.dataId)}readSync(e){let{values:t,dtype:n,complexTensorInfos:r,slice:i,shape:a,isPacked:o}=this.texData.get(e);if(i!=null){let t;t=o?new Rk(a,jk):new wk(a,jk);let r=this.runWebGLProgram(t,[{dataId:e,shape:a,dtype:n}],n),i=this.readSync(r.dataId);return this.disposeIntermediateTensorInfo(r),i}if(t!=null)return this.convertAndCacheOnCPU(e);if(n===`string`)return t;let s=this.activeTimers!=null,c;s&&(c=Wr());let l;return l=n===`complex64`?L_(this.readSync(r.real.dataId),this.readSync(r.imag.dataId)):this.getValuesFromTexture(e),s&&(this.downloadWaitMs+=Wr()-c),this.convertAndCacheOnCPU(e,l)}async read(e){if(this.pendingRead.has(e)){let t=this.pendingRead.get(e);return new Promise(e=>t.push(e))}let{values:t,shape:n,slice:r,dtype:i,complexTensorInfos:a,isPacked:o}=this.texData.get(e);if(r!=null){let t;t=o?new Rk(n,jk):new wk(n,jk);let r=this.runWebGLProgram(t,[{dataId:e,shape:n,dtype:i}],i),a=this.read(r.dataId);return this.disposeIntermediateTensorInfo(r),a}if(t!=null)return this.convertAndCacheOnCPU(e);if(N().getBool(`DEBUG`)&&!N().getBool(`WEBGL_DOWNLOAD_FLOAT_ENABLED`)&&N().getNumber(`WEBGL_VERSION`)===2)throw Error(`tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and WEBGL_VERSION=2 not yet supported.`);let s=null,c;if(i!==`complex64`&&N().get(`WEBGL_BUFFER_SUPPORTED`)){c=this.decode(e);let t=this.texData.get(c.dataId);s=this.gpgpu.createBufferFromTexture(t.texture.texture,...BT(n))}this.pendingRead.set(e,[]),i!==`complex64`&&await this.gpgpu.createAndWaitForFence();let l;if(i===`complex64`){let e=await Promise.all([this.read(a.real.dataId),this.read(a.imag.dataId)]),t=e[0],n=e[1];l=L_(t,n)}else if(s==null)l=this.getValuesFromTexture(e);else{let e=g(n);l=this.gpgpu.downloadFloat32MatrixFromBuffer(s,e)}if(c!=null&&this.disposeIntermediateTensorInfo(c),s!=null){let e=this.gpgpu.gl;J(e,()=>e.deleteBuffer(s))}let u=this.convertAndCacheOnCPU(e,l),d=this.pendingRead.get(e);return this.pendingRead.delete(e),d.forEach(e=>e(u)),this.pendingDisposal.has(e)&&(this.pendingDisposal.delete(e),this.disposeData(e)&&Bo().removeDataId(e,this),this.pendingDeletes--),u}readToGPU(e,t={}){let{values:n,shape:r,slice:i,dtype:a,isPacked:o,texture:s}=this.texData.get(e);if(a===`complex64`)throw Error(`Does not support reading texture for complex64 dtype.`);if(i!=null){let n;n=o?new Rk(r,jk):new wk(r,jk);let i=this.runWebGLProgram(n,[{dataId:e,shape:r,dtype:a}],a),s=this.readToGPU(i,t);return this.disposeIntermediateTensorInfo(i),s}if(s==null)throw Error(n==null?`There is no data on GPU or CPU.`:`Data is not on GPU but on CPU.`);let c=this.decode(e,t.customTexShape),l=Bo().makeTensorFromTensorInfo(c),u=this.texData.get(c.dataId);return Object.assign({tensorRef:l},u.texture)}bufferSync(e){let t=this.readSync(e.dataId);if(e.dtype===`string`)try{let n=t.map(e=>Kr(e));return L(e.shape,e.dtype,n)}catch{throw Error(`Failed to decode encoded string bytes into utf-8`)}return L(e.shape,e.dtype,t)}checkNumericalProblems(e){if(e!=null)for(let t=0;t<e.length;t++){let n=e[t];if(!qT(n))throw N().getBool(`WEBGL_RENDER_FLOAT32_CAPABLE`)?Error(`The value ${n} cannot be represented with your current settings. Consider enabling float32 rendering: 'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`):Error(`The value ${n} cannot be represented on this device.`)}}getValuesFromTexture(e){let{shape:t,dtype:n,isPacked:r}=this.texData.get(e),i=g(t);if(N().getBool(`WEBGL_DOWNLOAD_FLOAT_ENABLED`)){let n=this.decode(e),r=this.texData.get(n.dataId),a=this.gpgpu.downloadMatrixFromPackedTexture(r.texture.texture,...BT(t)).subarray(0,i);return this.disposeIntermediateTensorInfo(n),a}let a=N().getBool(`WEBGL_PACK`)&&r===!0,o=a?xE(t):t,s=a?new KD(o):new GD(o),c=this.runWebGLProgram(s,[{shape:o,dtype:n,dataId:e}],`float32`),l=this.texData.get(c.dataId),u=this.gpgpu.downloadByteEncodedFloatMatrixFromOutputTexture(l.texture.texture,l.texShape[0],l.texShape[1]).subarray(0,i);return this.disposeIntermediateTensorInfo(c),u}timerAvailable(){return N().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0}time(e){let t=this.activeTimers,n=[],r=!1;this.programTimersStack==null?(this.programTimersStack=n,r=!0):this.activeTimers.push(n),this.activeTimers=n,e();let i=h(this.activeTimers.map(e=>e.query)).filter(e=>e!=null),a=h(this.activeTimers.map(e=>e.name)).filter(e=>e!=null);this.activeTimers=t,r&&(this.programTimersStack=null);let o={uploadWaitMs:this.uploadWaitMs,downloadWaitMs:this.downloadWaitMs,kernelMs:null,wallMs:null};return(async()=>{if(N().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0){let e=await Promise.all(i);o.kernelMs=d(e),o.getExtraProfileInfo=()=>e.map((e,t)=>({name:a[t],ms:e})).map(e=>`${e.name}: ${e.ms}`).join(`, `)}else o.kernelMs={error:`WebGL query timers are not supported in this environment.`};return this.uploadWaitMs=0,this.downloadWaitMs=0,o})()}memory(){return{unreliable:!1,numBytesInGPU:this.numBytesInGPU,numBytesInGPUAllocated:this.textureManager.numBytesAllocated,numBytesInGPUFree:this.textureManager.numBytesFree}}startTimer(){return N().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0?this.gpgpu.beginQuery():{startMs:Wr(),endMs:null}}endTimer(e){return N().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0?(this.gpgpu.endQuery(),e):(e.endMs=Wr(),e)}async getQueryTime(e){if(N().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0)return this.gpgpu.waitForQueryAndGetTime(e);let t=e;return t.endMs-t.startMs}disposeData(e,t=!1){if(this.pendingDisposal.has(e))return!1;if(!this.texData.has(e))return!0;if(t?this.texData.get(e).refCount=0:this.texData.get(e).refCount--,!t&&this.texData.get(e).refCount>0)return!1;if(this.pendingRead.has(e))return this.pendingDisposal.add(e),this.pendingDeletes++,!1;this.releaseGPUData(e);let{complexTensorInfos:n}=this.texData.get(e);return n!=null&&(this.disposeData(n.real.dataId,t),this.disposeData(n.imag.dataId,t)),this.texData.delete(e),!0}releaseGPUData(e){let{texture:t,dtype:n,texShape:r,usage:i,isPacked:a,slice:o}=this.texData.get(e),s=o&&o.origDataId||e,c=this.dataRefCount.get(s);c>1?this.dataRefCount.set(s,c-1):(this.dataRefCount.delete(s),t!=null&&(this.numBytesInGPU-=this.computeBytes(r,n),this.textureManager.releaseTexture(t,r,i,a)));let l=this.texData.get(e);l.texture=null,l.texShape=null,l.isPacked=!1,l.slice=null}getTexture(e){return this.uploadToGPU(e),this.texData.get(e).texture.texture}getDataInfo(e){return this.texData.get(e)}shouldExecuteOnCPU(e,t=Gk){return N().getBool(`WEBGL_CPU_FORWARD`)&&e.every(e=>this.texData.get(e.dataId).texture==null&&g(e.shape)<t)}getGPGPUContext(){return this.gpgpu}where(e){mr(`tf.where() in webgl locks the UI thread. Call tf.whereAsync() instead`);let t=e.dataSync();return Bk(e.shape,t)}packedUnaryOp(e,t,n){let r=new Rk(e.shape,t),i=this.compileAndRun(r,[e],n);return Bo().makeTensorFromTensorInfo(i)}abs(e){if(this.shouldExecuteOnCPU([e])&&e.dtype!==`complex64`){let t=XO(this.texData.get(e.dataId).values);return this.makeOutput(e.shape,e.dtype,t)}if(N().getBool(`WEBGL_PACK_UNARY_OPERATIONS`))return this.packedUnaryOp(e,Dk,e.dtype);let t=new wk(e.shape,Dk),n=this.compileAndRun(t,[e]);return Bo().makeTensorFromTensorInfo(n)}makeTensorInfo(e,t,n){let r;if(t===`string`&&n!=null&&n.length>0&&ne(n[0])){let i=n.map(e=>Gr(e));r=this.write(i,e,t)}else r=this.write(n,e,t);return this.texData.get(r).usage=null,{dataId:r,shape:e,dtype:t}}makeOutput(e,t,n){return Bo().makeTensorFromTensorInfo(this.makeTensorInfo(e,t,n),this)}unpackTensor(e){let t=new zk(e.shape);return this.runWebGLProgram(t,[e],e.dtype)}packTensor(e){let t=new mk(e.shape);return this.runWebGLProgram(t,[e],e.dtype,null,!0)}packedReshape(e,t){let n=[yE(e.shape),...bE(e.shape)],r={dtype:e.dtype,shape:n,dataId:e.dataId},i=new hk([yE(t),...bE(t)],n),a=[n],o=this.runWebGLProgram(i,[r],e.dtype,a,!0);return{dataId:o.dataId,shape:t,dtype:o.dtype}}decode(e,t){let{isPacked:n,shape:r,dtype:i}=this.texData.get(e);t!=null&&f(g(r)<=t[0]*t[1]*4,()=>`customTexShape is too small. Row * Column * 4 should be equal or larger than the size of the tensor data.`);let a=xE(r),o;o=n?new WD(a):new UD(a);let s=[t??BT(a)];return{dtype:i,shape:r,dataId:this.runWebGLProgram(o,[{shape:a,dtype:i,dataId:e}],i,s,!0,t).dataId}}runWebGLProgram(e,t,n,r,i=!1,a){let o=this.makeTensorInfo(e.outputShape,n),s=this.texData.get(o.dataId);if(e.packedOutput&&(s.isPacked=!0),e.outPackingScheme===FT.DENSE&&(s.texShape=(a??BT(e.outputShape)).map(e=>e*2)),e.outTexUsage!=null&&(s.usage=e.outTexUsage),g(o.shape)===0)return s.values=T(o.dtype,0),o;let c=[],l=t.map(t=>{if(t.dtype===`complex64`)throw Error(`GPGPUProgram does not support complex64 input. For complex64 dtypes, please separate the program into real and imaginary parts.`);let n=this.texData.get(t.dataId);if(n.texture==null){if(!e.packedInputs&&g(t.shape)<=N().getNumber(`WEBGL_SIZE_UPLOAD_UNIFORM`))return{shape:t.shape,texData:null,isUniform:!0,uniformValues:n.values};e.packedInputs&&(n.isPacked=!0,n.shape=t.shape)}if(this.uploadToGPU(t.dataId),!!n.isPacked!=!!e.packedInputs)t=n.isPacked?this.unpackTensor(t):this.packTensor(t),c.push(t),n=this.texData.get(t.dataId);else if(n.isPacked&&!wE(n.shape,t.shape)){let e=t,r=t.shape;t.shape=n.shape,t=this.packedReshape(t,r),c.push(t),n=this.texData.get(t.dataId),e.shape=r}return{shape:t.shape,texData:n,isUniform:!1}});this.uploadToGPU(o.dataId);let u={shape:o.shape,texData:s,isUniform:!1},d=VD(e,l,u),f=this.getAndSaveBinary(d,()=>LD(this.gpgpu,e,l,u)),p=this.activeTimers!=null,m;p&&(m=this.startTimer()),N().get(`ENGINE_COMPILE_ONLY`)||BD(this.gpgpu,f,l,u,r),c.forEach(e=>this.disposeIntermediateTensorInfo(e)),p&&(m=this.endTimer(m),this.activeTimers.push({name:e.constructor.name,query:this.getQueryTime(m)}));let h=N().get(`WEBGL_FLUSH_THRESHOLD`);if(h>0){let e=Wr();e-this.lastGlFlushTime>h&&(this.gpgpu.gl.flush(),this.lastGlFlushTime=e)}if(!N().getBool(`WEBGL_LAZILY_UNPACK`)&&s.isPacked&&i===!1){let e=this.unpackTensor(o);return this.disposeIntermediateTensorInfo(o),e}return o}compileAndRun(e,t,n,r,i=!1){return n||=t[0].dtype,this.runWebGLProgram(e,t,n,r,i)}getAndSaveBinary(e,t){return e in this.binaryCache||(this.binaryCache[e]=t()),this.binaryCache[e]}getTextureManager(){return this.textureManager}dispose(){this.disposed||=(N().getBool(`IS_TEST`)||Object.keys(this.binaryCache).forEach(e=>{this.gpgpu.deleteProgram(this.binaryCache[e].webGLProgram),delete this.binaryCache[e]}),this.textureManager.dispose(),this.canvas!=null&&typeof HTMLCanvasElement<`u`&&this.canvas instanceof HTMLCanvasElement?this.canvas.remove():this.canvas=null,this.gpgpuCreatedLocally&&(this.gpgpu.program=null,this.gpgpu.dispose()),!0)}floatPrecision(){return this.floatPrecisionValue??=Vo(()=>{if(!N().get(`WEBGL_RENDER_FLOAT32_ENABLED`)){let e=N().getBool(`DEBUG`);N().set(`DEBUG`,!1);let t=this.abs(U(1e-8)).dataSync()[0];if(N().set(`DEBUG`,e),t>0)return 32}return 16}),this.floatPrecisionValue}epsilon(){return this.floatPrecision()===32?Vk:Hk}uploadToGPU(e){let t=this.texData.get(e),{shape:n,dtype:r,values:i,texture:a,usage:o,isPacked:s}=t;if(a!=null)return;let c=this.activeTimers!=null,l;c&&(l=Wr());let u=t.texShape;if(u??(u=SE(n,s),t.texShape=u),i!=null){let e=xE(n),a,o=u[1],d=u[0],f=i instanceof Uint8Array||i instanceof Uint8ClampedArray;(s||!f)&&([o,d]=VT(u[0],u[1])),a=s?new JD(e,f):new qD(e,f);let p=f?[d,o]:u,m=this.makeTensorInfo(p,r),h=this.texData.get(m.dataId);h.usage=f?IT.PIXELS:IT.UPLOAD,h.texShape=p,this.gpgpu.uploadDenseMatrixToTexture(this.getTexture(m.dataId),o,d,i);let g=[[d,o]],_=this.runWebGLProgram(a,[m],r,g,!0),v=this.texData.get(_.dataId);t.texShape=v.texShape,t.isPacked=v.isPacked,t.usage=v.usage,N().get(`ENGINE_COMPILE_ONLY`)?this.disposeData(_.dataId):(t.texture=v.texture,t.values=null,this.texData.delete(_.dataId)),this.disposeIntermediateTensorInfo(m),c&&(this.uploadWaitMs+=Wr()-l)}else t.texture=this.acquireTexture(u,o,r,s)}convertAndCacheOnCPU(e,t){let n=this.texData.get(e),{dtype:r}=n;return this.releaseGPUData(e),t!=null&&(n.values=Yk(t,r)),n.values}acquireTexture(e,t,n,r){if(this.numBytesInGPU+=this.computeBytes(e,n),!this.warnedAboutMemory&&this.numBytesInGPU>this.numMBBeforeWarning*1024*1024){let e=(this.numBytesInGPU/1024/1024).toFixed(2);this.warnedAboutMemory=!0,console.warn(`High memory usage in GPU: ${e} MB, most likely due to a memory leak`)}return this.textureManager.acquireTexture(e,t,r)}computeBytes(e,t){return e[0]*e[1]*A(t)}checkCompileCompletion(){for(let[,e]of Object.entries(this.binaryCache))this.checkCompletion_(e)}async checkCompileCompletionAsync(){let e=[];if(this.gpgpu.parallelCompilationExtension){for(let[,t]of Object.entries(this.binaryCache))e.push(this.checkCompletionAsync_(t));return Promise.all(e)}for(let[,t]of Object.entries(this.binaryCache)){let n=new Promise(e=>{try{this.checkCompletion_(t),e(!0)}catch(e){throw e}});e.push(n)}return Promise.all(e)}async checkCompletionAsync_(e){return this.gpgpu.gl.getProgramParameter(e.webGLProgram,this.gpgpu.parallelCompilationExtension.COMPLETION_STATUS_KHR)?this.checkCompletion_(e):(await m_(),this.checkCompletionAsync_(e))}checkCompletion_(e){if(this.gpgpu.gl.getProgramParameter(e.webGLProgram,this.gpgpu.gl.LINK_STATUS)===!1)throw console.log(this.gpgpu.gl.getProgramInfoLog(e.webGLProgram)),this.gpgpu.gl.getShaderParameter(e.fragmentShader,this.gpgpu.gl.COMPILE_STATUS)===!1?($T(e.source,this.gpgpu.gl.getShaderInfoLog(e.fragmentShader)),Error(`Failed to compile fragment shader.`)):Error(`Failed to link vertex and fragment shaders.`);return!0}getUniformLocations(){for(let[,e]of Object.entries(this.binaryCache)){let{uniformLocations:t,customUniformLocations:n,infLoc:r,nanLoc:i,inShapesLocations:a,inTexShapesLocations:o,outShapeLocation:s,outShapeStridesLocation:c,outTexShapeLocation:l}=RD(this.gpgpu,e.program,e.webGLProgram);e.uniformLocations=t,e.customUniformLocations=n,e.infLoc=r,e.nanLoc=i,e.inShapesLocations=a,e.inTexShapesLocations=o,e.outShapeLocation=s,e.outShapeStridesLocation=c,e.outTexShapeLocation=l}}};Jk.nextDataId=0;function Yk(e,t){if(t===`float32`||t===`complex64`)return e;if(t===`int32`||t===`bool`){let n=t===`int32`?new Int32Array(e.length):new Uint8Array(e.length);for(let t=0;t<n.length;++t)n[t]=Math.round(e[t]);return n}throw Error(`Unknown dtype ${t}`)}Ii()&&Jo(`webgl`,()=>new Jk,2);var Xk=`
  if (isnan(a)) return a;
  if (isnan(b)) return b;
`,Zk=class{constructor(e,t,n){this.variableNames=[`A`,`B`],this.outputShape=z(t,n),this.enableShapeUniforms=HD(this.outputShape.length),this.userCode=`
      float binaryOperation(float a, float b) {
        ${e}
      }

      void main() {
        float a = getAAtOutCoords();
        float b = getBAtOutCoords();
        setOutput(binaryOperation(a, b));
      }
    `}},Qk=`
  result.r = isNaN.r ? NAN : result.r;
  result.g = isNaN.g ? NAN : result.g;
  result.b = isNaN.b ? NAN : result.b;
  result.a = isNaN.a ? NAN : result.a;
`,$k=class{constructor(e,t,n,r=!1){this.variableNames=[`A`,`B`],this.supportsBroadcasting=!0,this.packedInputs=!0,this.packedOutput=!0,this.outputShape=z(t,n);let i=this.outputShape.length;this.enableShapeUniforms=HD(i);let a=``;if(r){if(i===0||g(this.outputShape)===1)a=`
          result.y = 0.;
          result.z = 0.;
          result.w = 0.;
        `;else if(a=`
          ${X(i)} coords = getOutputCoords();
        `,i===1)this.enableShapeUniforms?a+=`
            result.y = (coords + 1) >= outShape ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `:a+=`
            result.y = (coords + 1) >= ${this.outputShape[0]} ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `;else{let e=fk(`coords`,i);this.enableShapeUniforms?a+=`
            bool nextRowOutOfBounds =
              (${e[i-2]} + 1) >= outShape[${i} - 2];
            bool nextColOutOfBounds =
              (${e[i-1]} + 1) >= outShape[${i} - 1];
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `:a+=`
            bool nextRowOutOfBounds =
              (${e[i-2]} + 1) >= ${this.outputShape[i-2]};
            bool nextColOutOfBounds =
              (${e[i-1]} + 1) >= ${this.outputShape[i-1]};
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `}}this.userCode=`
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${e}
      }

      void main() {
        vec4 a = getAAtOutCoords();
        vec4 b = getBAtOutCoords();

        vec4 result = binaryOperation(a, b);
        ${a}

        setOutput(result);
      }
    `}};function eA(e){let{inputs:t,backend:n}=e,{x:r}=t;return n.incRef(r.dataId),{dataId:r.dataId,shape:r.shape,dtype:r.dtype}}var tA={kernelName:jt,backendName:`webgl`,kernelFunc:eA};function nA(e){let{inputs:t,backend:n}=e,{real:r,imag:i}=t,a=n.makeTensorInfo(r.shape,`complex64`),o=n.texData.get(a.dataId);return o.complexTensorInfos={real:eA({inputs:{x:r},backend:n}),imag:eA({inputs:{x:i},backend:n})},a}var rA={kernelName:Je,backendName:`webgl`,kernelFunc:nA},iA=`return (a < 0.) ? b * a : a;`,aA=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function oA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{alpha:a}=r,o=n.makeTensorInfo([],`float32`,Vr(a,`float32`)),s=N().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)?new $k(aA,i.shape,o.shape):new Zk(iA,i.shape,o.shape),c=n.runWebGLProgram(s,[i,o],`float32`);return n.disposeIntermediateTensorInfo(o),c}var sA={kernelName:Lt,backendName:`webgl`,kernelFunc:oA},cA=`return (a < 0.) ? b * a : a;`,lA=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function uA(e){let{inputs:t,backend:n}=e,{x:r,alpha:i}=t,a=N().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)?new $k(lA,r.shape,i.shape):new Zk(cA,r.shape,i.shape);return n.runWebGLProgram(a,[r,i],`float32`)}var dA={kernelName:fn,backendName:`webgl`,kernelFunc:uA},fA=`if (isnan(x)) return x;`;function Z({opSnippet:e,packedOpSnippet:t,cpuKernelImpl:n,dtype:r}){return({inputs:i,backend:a})=>{let{x:o}=i,s=a,c=r||o.dtype;if(s.shouldExecuteOnCPU([o])&&n!=null){let e=n(s.texData.get(o.dataId).values,c);return s.makeTensorInfo(o.shape,c,e)}let l=N().getBool(`WEBGL_PACK_UNARY_OPERATIONS`)&&t!=null,u;return u=l?new Rk(o.shape,t):new wk(o.shape,e),s.runWebGLProgram(u,[o],c)}}function pA({opSnippet:e,packedOpSnippet:t,checkOutOfBounds:n=!1,supportsComplex:r=!1,cpuKernelImpl:i,dtype:a}){return({inputs:o,backend:s})=>{let{a:c,b:l}=o,u=s;if(r&&c.dtype===`complex64`){let t=u.texData.get(c.dataId),n=u.texData.get(l.dataId),[r,i]=[[t.complexTensorInfos.real,n.complexTensorInfos.real],[t.complexTensorInfos.imag,n.complexTensorInfos.imag]].map(t=>{let[n,r]=t,i={dataId:n.dataId,dtype:n.dtype,shape:c.shape},a={dataId:r.dataId,dtype:r.dtype,shape:l.shape},o=new Zk(e,c.shape,l.shape);return u.runWebGLProgram(o,[i,a],xi(n.dtype,r.dtype))}),a=nA({inputs:{real:r,imag:i},backend:u});return u.disposeIntermediateTensorInfo(r),u.disposeIntermediateTensorInfo(i),a}let d=a||xi(c.dtype,l.dtype);if((c.dtype===`string`||l.dtype===`string`||u.shouldExecuteOnCPU([c,l]))&&i!=null){let e=u.texData.get(c.dataId).values,t=u.texData.get(l.dataId).values,n=c.dtype===`string`?yv(e):e,r=c.dtype===`string`?yv(t):t,[a,o]=i(c.shape,l.shape,n,r,d),s=u.makeTensorInfo(o,d),f=u.texData.get(s.dataId);return f.values=a,s}let f=N().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)&&t!=null,p;return p=f?new $k(t,c.shape,l.shape,n):new Zk(e,c.shape,l.shape),u.runWebGLProgram(p,[c,l],d)}}function mA(e,t=!1){if(e===`linear`)return t?Nk:Ek;if(e===`relu`)return t?Fk:kk;if(e===`elu`)return t?Pk:Ok;if(e===`relu6`)return t?Ik:Ak;if(e===`prelu`)return t?lA:cA;if(e===`leakyrelu`)return t?aA:iA;if(e===`sigmoid`)return t?Lk:Mk;throw Error(`Activation ${e} has not been implemented for the WebGL backend.`)}var hA=class{constructor(e,t,n,r=!1,i=!1,a=!1,o=null,s=!1,c=!1){this.variableNames=[`matrixA`,`matrixB`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=n,this.enableShapeUniforms=HD(this.outputShape.length);let l=r?e[1]:e[2],u=Math.ceil(l/2),d=r?`i * 2, rc.y`:`rc.y, i * 2`,f=i?`rc.z, i * 2`:`i * 2, rc.z`,p=r?[`a.xxyy`,`a.zzww`]:[`a.xxzz`,`a.yyww`],m=i?[`b.xzxz`,`b.ywyw`]:[`b.xyxy`,`b.zwzw`],h=``,g=``;o&&(h=s?`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${o}
        }`:c?`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${o}
        }`:`vec4 activation(vec4 x) {
          ${o}
        }`,g=`result = activation(result);`);let _=a?`result += getBiasAtOutCoords();`:``;a&&this.variableNames.push(`bias`),s&&this.variableNames.push(`preluActivationWeights`),c&&this.variableNames.push(`leakyreluAlpha`);let v=`rc.x`,y=`rc.x`;e[0]<t[0]?v=`int(min(float(rc.x), ${e[0]-1}.))`:t[0]<e[0]&&(y=`int(min(float(rc.x), ${t[0]-1}.))`),this.userCode=`
      ${h}
      // Don't use uniform for sharedDimensionPacked for performance.
      const float sharedDimension = ${u}.0;

      vec4 dot2x2ARowBCol(ivec3 rc) {
        vec4 result = vec4(0);
        for (int i = 0; i < ${u}; i++) {
          int batchA = ${v};
          int batchB = ${y};
          vec4 a = getMatrixA(batchA, ${d});
          vec4 b = getMatrixB(batchB, ${f});

          // These swizzled products need to be separately added.
          // See: https://github.com/tensorflow/tfjs/issues/1735
          result += (${p[0]} * ${m[0]});
          result += (${p[1]} * ${m[1]});
        }
        return result;
      }

      void main() {
        ivec3 rc = getOutputCoords();
        vec4 result = dot2x2ARowBCol(rc);

        ${_}

        ${g}

        setOutput(result);
      }
    `}},gA={REAL:`return areal * breal - aimag * bimag;`,IMAG:`return areal * bimag + aimag * breal;`},_A=class{constructor(e,t,n){this.variableNames=[`AReal`,`AImag`,`BReal`,`BImag`],this.outputShape=z(t,n),this.userCode=`
      float binaryOpComplex(
          float areal, float aimag, float breal, float bimag) {
        ${e}
      }

      void main() {
        float areal = getARealAtOutCoords();
        float aimag = getAImagAtOutCoords();
        float breal = getBRealAtOutCoords();
        float bimag = getBImagAtOutCoords();
        setOutput(binaryOpComplex(areal, aimag, breal, bimag));
      }
    `}},vA=`return a * b;`;function yA(e){let{inputs:t,backend:n}=e,{a:r,b:i}=t,a=xi(r.dtype,i.dtype);if(r.dtype===`complex64`){let e=n.texData.get(r.dataId),t=n.texData.get(i.dataId),a=new _A(gA.REAL,r.shape,i.shape),o=new _A(gA.IMAG,r.shape,i.shape),s=[{dataId:e.complexTensorInfos.real.dataId,dtype:e.complexTensorInfos.real.dtype,shape:r.shape},{dataId:e.complexTensorInfos.imag.dataId,dtype:e.complexTensorInfos.imag.dtype,shape:r.shape},{dataId:t.complexTensorInfos.real.dataId,dtype:t.complexTensorInfos.real.dtype,shape:i.shape},{dataId:t.complexTensorInfos.imag.dataId,dtype:t.complexTensorInfos.imag.dtype,shape:i.shape}],c=n.runWebGLProgram(a,s,`float32`),l=n.runWebGLProgram(o,s,`float32`),u=nA({inputs:{real:c,imag:l},backend:n});return n.disposeIntermediateTensorInfo(c),n.disposeIntermediateTensorInfo(l),u}if(n.shouldExecuteOnCPU([r,i])){let e=n.texData.get(r.dataId),t=n.texData.get(i.dataId),[o,s]=BO(r.shape,i.shape,e.values,t.values,a),c=n.makeTensorInfo(s,a),l=n.texData.get(c.dataId);return l.values=o,c}let o;return o=N().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)?new $k(vA,r.shape,i.shape):new Zk(vA,r.shape,i.shape),n.runWebGLProgram(o,[r,i],a)}var bA={kernelName:nn,backendName:`webgl`,kernelFunc:yA};function xA(e,t,n){let r=[yE(e.shape),...bE(e.shape)],i={dtype:e.dtype,shape:r,dataId:e.dataId},a=new hk([yE(t),...bE(t)],r),o=[r],s=n.runWebGLProgram(a,[i],e.dtype,o,!0);return{dataId:s.dataId,shape:t,dtype:s.dtype}}function Q(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{shape:a}=r,o=n,s=g(i.shape),c=S(a,s),l=g(c);f(s===l,()=>`The new shape (${c}) has ${l} elements and the old shape (${i.shape}) has ${s} elements. The new shape and old shape must have the same number of elements.`);let u=o.texData.get(i.dataId);return u.isPacked&&!wE(i.shape,c)&&!(u.texture!==null&&wE(u.shape,c))?xA(i,c,o):(o.incRef(i.dataId),{dataId:i.dataId,shape:c,dtype:i.dtype})}var SA={kernelName:bn,backendName:`webgl`,kernelFunc:Q},CA=class{constructor(e,t){this.variableNames=[`x`];let{windowSize:n,batchSize:r,inSize:i,outSize:a}=e;this.outputShape=[r,a];let o=Math.floor(n/4)*4,s=n%4,c=`sumValue += dot(values, ones);`;if(t!=null){let e=1/t;c=`sumValue += dot(values * ${v(e)?e.toPrecision(2):e}, ones);`}let l=``;i%n>0&&(l=`
        if (inIdx < 0 || inIdx >= ${i}) {
          return 0.0;
        }
      `),this.userCode=`
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${l}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${n};

        float sumValue = 0.0;

        for (int i = 0; i < ${o}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${c}
        }

        int inIdx = inOffset + ${o};
        if (${s===1}) {
          vec4 values = vec4(getValue(batch, inIdx), 0.0, 0.0, 0.0);

          ${c}
        } else if (${s===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1), 0.0, 0.0);

          ${c}
        } else if (${s===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2), 0.0);

          ${c}
        }
        setOutput(sumValue);
      }
    `}},wA=class{constructor(e,t){this.variableNames=[`x`];let{windowSize:n,batchSize:r,inSize:i,outSize:a}=e;this.outputShape=[r,a];let o=`0.0`,s=``;t===`prod`?o=`1.0`:t===`min`?(o=`1.0 / 1e-20`,s=`min`):t===`max`&&(o=`-1.0 / 1e-20`,s=`max`);let c=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t===`sum`?c=`sumValue`:t===`prod`?c=`prodValue`:t===`all`?c=`allValue`:t===`any`&&(c=`anyValue`);let l=Math.floor(n/4)*4,u=n%4,d=`
      if (${t===`sum`}) {
        sumValue += dot(values, ones);
      } else if (${t===`prod`}) {
        vec2 tmp = vec2(values[0], values[1]) * vec2(values[2], values[3]);
        prodValue *= tmp[0] * tmp[1];
      } else {
        minMaxValue = ${s}(values, minMaxValue);
        if (${t===`min`} || ${t===`max`}) {
          minMaxValue = ${s}(values, minMaxValue);
          bvec4 isNaN = isnan(values);
          if (isNaN.r || isNaN.g || isNaN.b || isNaN.a) {
            minMaxValue = vec4(NAN);
          }
        }
      }
    `,f=`vec4`;t===`all`?(o=`1.0`,d=`
        bool reducedAllValue = all(values);
        float floatedReducedAllValue = float(reducedAllValue);
        allValue = float(allValue >= 1.0 && floatedReducedAllValue >= 1.0);
      `,f=`bvec4`):t===`any`&&(o=`0.0`,d=`
        bool reducedAnyValue = any(values);
        float floatedReducedAnyValue = float(reducedAnyValue);
        anyValue = float(anyValue >= 1.0 || floatedReducedAnyValue >= 1.0);
      `,f=`bvec4`);let p=``;i%n>0&&(p=`
        if (inIdx < 0 || inIdx >= ${i}) {
          return initializationValue;
        }
      `),this.userCode=`
      const float initializationValue = ${o};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${p}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${n};

        vec4 minMaxValue = vec4(${o});
        float prodValue = 1.0;
        float sumValue = 0.0;
        float allValue = 1.0;
        float anyValue = 0.0;

        for (int i = 0; i < ${l}; i += 4) {
          int inIdx = inOffset + i;
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${d}
        }

        int inIdx = inOffset + ${l};
        if (${u===1}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          ${d}
        } else if (${u===2}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          ${d}
        } else if (${u===3}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          ${d}
        }
        setOutput(${c});
      }
    `}};function TA(e){let t=[];for(;t.length===0||t[t.length-1].outSize!==1;){let n=t.length?t[t.length-1].outSize:e[1],r=S_(n);t.push({inSize:n,windowSize:r,outSize:Math.ceil(n/r)})}return t}function EA(e,t,n,r){let i=TA(e.shape),a=e;for(let o=0;o<i.length;o++){let{inSize:s,windowSize:c,outSize:l}=i[o],u,d;u=n===`mean`?o===0?new CA({windowSize:c,inSize:s,batchSize:e.shape[0],outSize:l},s):new CA({windowSize:c,inSize:s,batchSize:e.shape[0],outSize:l}):new wA({windowSize:c,inSize:s,batchSize:e.shape[0],outSize:l},n),d=a,a=r.runWebGLProgram(u,[a],t),d.dataId!==e.dataId&&r.disposeIntermediateTensorInfo(d)}return a}var DA=class{constructor(e,t){this.variableNames=[`A`];let n=Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[t[r]];this.outputShape=n,this.rank=n.length;let r=X(this.rank),i=OA(t);this.userCode=`
    void main() {
      ${r} resRC = getOutputCoords();
      setOutput(getA(${i}));
    }
    `}};function OA(e){let t=e.length;if(t>6)throw Error(`Transpose for rank ${t} is not yet supported`);let n=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`,`resRC.u`,`resRC.v`],r=Array(t);for(let t=0;t<e.length;t++)r[e[t]]=n[t];return r.join()}var kA=class{constructor(e,t){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0;let n=Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[t[r]];if(this.outputShape=n,this.rank=n.length,this.rank>6)throw Error(`Packed transpose for rank ${this.rank} is not yet supported.`);let r=X(this.rank),i=dk(`rc`,this.rank),a=Array(this.rank);for(let e=0;e<t.length;e++)a[t[e]]=i[e];let o=`vec2(${a.slice(-2).join()})`,s=`++${i[this.rank-1]} < ${n[this.rank-1]}`,c=`getChannel(getA(${a.join()}), ${o})`;this.userCode=`
    void main() {
      ${r} rc = getOutputCoords();
      vec4 result = vec4(0.);
      result[0] = ${c};
      if(${s}) {
        result[1] = ${c};
      }
      --${i[this.rank-1]};
      if(++${i[this.rank-2]} < ${n[this.rank-2]}) {
        result[2] = ${c};
        if(${s}) {
          result[3] = ${c};
        }
      }
      setOutput(result);
    }
    `}};function AA(e,t,n){let r=N().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new kA(e.shape,t):new DA(e.shape,t);return n.runWebGLProgram(r,[e],e.dtype)}function jA(e,t,n,r){let i=t,a=e.shape.length,o=C(i,e.shape),s=o,c=_u(s,a),l=c!=null,u=e;l&&(u=AA(e,c,r),s=yu(s.length,a)),gu(`sum`,s,a);let[d,f]=mu(u.shape,s),p=d;n&&(p=hu(d,o));let m=g(f),h=g(e.shape)/m,_=Q({inputs:{x:u},attrs:{shape:[h,m]},backend:r}),v=EA(_,Si(e.dtype),`sum`,r),y=Q({inputs:{x:v},attrs:{shape:p},backend:r});return r.disposeIntermediateTensorInfo(_),r.disposeIntermediateTensorInfo(v),l&&r.disposeIntermediateTensorInfo(u),y}function MA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;return jA(i,a,o,n)}var NA={kernelName:`Sum`,backendName:`webgl`,kernelFunc:MA};function PA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{perm:a}=r,o=n,s=i.shape.length,c=Array(s);for(let e=0;e<c.length;e++)c[e]=i.shape[a[e]];let l;if(o.shouldExecuteOnCPU([i])){let e=o.texData.get(i.dataId).values,t=lk(e,i.shape,i.dtype,a,c);l=o.makeTensorInfo(c,i.dtype);let n=o.texData.get(l.dataId);n.values=t}else l=AA(i,a,o);return l}var FA={kernelName:rr,backendName:`webgl`,kernelFunc:PA};function IA({a:e,b:t,transposeA:n,transposeB:r,backend:i,bias:a=null,preluActivationWeights:o=null,leakyreluAlpha:s=0,activation:c=null}){let l=e.shape.length,u=t.shape.length,d=n?e.shape[l-2]:e.shape[l-1],p=r?t.shape[u-1]:t.shape[u-2],m=n?e.shape[l-1]:e.shape[l-2],h=r?t.shape[u-2]:t.shape[u-1],_=e.shape.slice(0,-2),v=t.shape.slice(0,-2),y=g(_),b=g(v),x=z(e.shape.slice(0,-2),t.shape.slice(0,-2)).concat([m,h]);f(d===p,()=>`Error in matMul: inner shapes (${d}) and (${p}) of Tensors with shapes ${e.shape} and ${t.shape} and transposeA=${n} and transposeB=${r} must match.`);let S=n?[y,d,m]:[y,m,d],C=r?[b,h,p]:[b,p,h],w=Q({inputs:{x:e},backend:i,attrs:{shape:S}}),T=Q({inputs:{x:t},backend:i,attrs:{shape:C}}),E=[w,T],D=Math.max(y,b),O=n?w.shape[1]:w.shape[2],ee=a!=null,k=o!=null,A=c===`leakyrelu`,te=c==null?null:mA(c,!0),ne=ee||k||A||te!=null,re;if((m===1||h===1)&&O>1e3&&ne===!1){let e=w,t=T;n&&(e=PA({inputs:{x:w},backend:i,attrs:{perm:[0,2,1]}}),E.push(e)),r&&(t=PA({inputs:{x:T},backend:i,attrs:{perm:[0,2,1]}}),E.push(t));let a=h!==1,o=h===1,s=e;a&&(s=Q({inputs:{x:e},backend:i,attrs:{shape:[D,O,1]}}),E.push(s));let c=h===1?2:1,l=t;o&&(l=Q({inputs:{x:t},backend:i,attrs:{shape:[D,1,O]}}),E.push(l));let u=yA({inputs:{a:s,b:l},backend:i});re=MA({inputs:{x:u},backend:i,attrs:{axis:c,keepDims:!0}}),E.push(u)}else{let c=xi(e.dtype,t.dtype),l=new hA(S,C,[D,m,h],n,r,ee,te,k,A),u=[w,T];if(a!=null&&u.push(a),k&&u.push(o),A){let e=i.makeTensorInfo([],`float32`,Vr(s,`float32`));u.push(e),E.push(e)}re=i.runWebGLProgram(l,u,c)}let ie=Q({inputs:{x:re},backend:i,attrs:{shape:x}});E.push(re);for(let e of E)i.disposeIntermediateTensorInfo(e);return ie}function LA(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a,bias:o,preluActivationWeights:s}=t,{transposeA:c,transposeB:l,activation:u,leakyreluAlpha:d}=r;return IA({a:i,b:a,transposeA:c,transposeB:l,backend:n,bias:o,preluActivationWeights:s,leakyreluAlpha:d,activation:u})}var RA={kernelName:dr,backendName:`webgl`,kernelFunc:LA},zA=`return abs(x);`;function BA(e){let{inputs:t,backend:n}=e,{x:r}=t;if(n.shouldExecuteOnCPU([r])&&r.dtype!==`complex64`){let e=XO(n.texData.get(r.dataId).values);return n.makeTensorInfo(r.shape,r.dtype,e)}let i;return i=N().getBool(`WEBGL_PACK_UNARY_OPERATIONS`)?new Rk(r.shape,zA):new wk(r.shape,zA),n.runWebGLProgram(i,[r],r.dtype)}var VA={kernelName:`Abs`,backendName:`webgl`,kernelFunc:BA},HA={kernelName:De,backendName:`webgl`,kernelFunc:Z({opSnippet:Tk+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return acos(x);
`})},UA={kernelName:Oe,backendName:`webgl`,kernelFunc:Z({opSnippet:Tk+`
  if (x < 1.0) return NAN;
return log(x + sqrt(x * x - 1.0));`})},WA=`return a + b;`,GA={kernelName:`Add`,backendName:`webgl`,kernelFunc:pA({opSnippet:WA,packedOpSnippet:WA,supportsComplex:!0,cpuKernelImpl:yO})},KA=class{constructor(e,t){this.outputShape=[],this.outputShape=e,this.variableNames=t.map((e,t)=>`T${t}`);let n=[];this.variableNames.forEach(e=>{n.push(`float v${e} = get${e}AtOutCoords();`)});let r=this.variableNames.map(e=>`v${e}`).join(` + `);this.userCode=`
      void main() {
        ${n.join(`
        `)}

        float result = ${r};
        setOutput(result);
      }
    `}},qA=class{constructor(e,t){this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.variableNames=t.map((e,t)=>`T${t}`);let n=[];this.variableNames.forEach(e=>{n.push(`vec4 v${e} = get${e}AtOutCoords();`)});let r=this.variableNames.map(e=>`v${e}`).join(` + `);this.userCode=`
      void main() {
        ${n.join(`
        `)}

        vec4 result = ${r};
        setOutput(result);
      }
    `}};function JA(e){let{inputs:t,backend:n}=e,r=t;if(r.length===1)return eA({inputs:{x:r[0]},backend:n});if(r.length>N().get(`WEBGL_MAX_TEXTURES_IN_SHADER`)){let e=Math.floor(r.length/2);return JA({inputs:[JA({inputs:r.slice(0,e),backend:n}),JA({inputs:r.slice(e),backend:n})],backend:n})}let i=r.map(e=>e.dtype).reduce((e,t)=>xi(e,t)),a=r.map(e=>e.shape),o=N().getBool(`WEBGL_PACK`)?new qA(r[0].shape,a):new KA(r[0].shape,a);return n.runWebGLProgram(o,r,i)}var YA={kernelName:ke,backendName:`webgl`,kernelFunc:JA};function XA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=i.shape.length,c=C(a,i.shape),l=c,u=_u(l,s),d=i;u!=null&&(d=PA({inputs:{x:i},backend:n,attrs:{perm:u}}),l=yu(l.length,s)),gu(`all`,l,s);let[f,p]=mu(d.shape,l),m=g(p),h=Q({inputs:{x:d},backend:n,attrs:{shape:[-1,m]}}),_=EA(h,h.dtype,`all`,n),v;if(o){let e=hu(f,c);v=Q({inputs:{x:_},backend:n,attrs:{shape:e}})}else v=Q({inputs:{x:_},backend:n,attrs:{shape:f}});return n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(_),u!=null&&n.disposeIntermediateTensorInfo(d),v}var ZA={kernelName:`All`,backendName:`webgl`,kernelFunc:XA};function QA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=i.shape.length,c=C(a,i.shape),l=c,u=_u(l,s),d=i;u!=null&&(d=PA({inputs:{x:i},backend:n,attrs:{perm:u}}),l=yu(l.length,s)),gu(`any`,l,s);let[f,p]=mu(d.shape,l),m=g(p),h=Q({inputs:{x:d},backend:n,attrs:{shape:[-1,m]}}),_=EA(h,h.dtype,`any`,n),v;if(o){let e=hu(f,c);v=Q({inputs:{x:_},backend:n,attrs:{shape:e}})}else v=Q({inputs:{x:_},backend:n,attrs:{shape:f}});return n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(_),u!=null&&n.disposeIntermediateTensorInfo(d),v}var $A={kernelName:`Any`,backendName:`webgl`,kernelFunc:QA},ej=class{constructor(e,t,n){this.variableNames=[`A`];let{windowSize:r,batchSize:i,outSize:a}=e;n||this.variableNames.push(`bestIndicesA`),this.outputShape=[i,a];let o=t===`max`?`>`:`<`,s=n?`inOffset + i;`:`round(getBestIndicesA(batch, inOffset + i));`;this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${r};

        int bestIndex = inOffset;
        float bestValue = getA(batch, bestIndex);

        for (int i = 0; i < ${r}; i++) {
          int inIdx = ${s};
          float candidate = getA(batch, inIdx);
          if (candidate ${o} bestValue) {
            bestValue = candidate;
            bestIndex = inIdx;
          }
        }
        setOutput(float(bestIndex));
      }
    `}},tj=class{constructor(e,t,n,r){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,f(e.length>2,()=>`Packed arg${n.charAt(0).toUpperCase()+n.slice(1)} supports only inputs with rank above 2.`);let i=e[e.length-1],a=Math.ceil(i/t);this.outputShape=e.slice(0,-1),a>1&&this.outputShape.push(a),r||this.variableNames.push(`bestIndicesA`);let o=this.outputShape,s=o.length,c=X(s),l=fk(`coords`,s),u,d;if(a===1){d=s+1;let e=X(d);u=`
        ${e} sourceLocR = ${e}(${l.join()}, 0);
        ++${l[s-1]};
        ${e} sourceLocG = ${e}(${l.join()}, 0);
        ++${l[s-2]};
        ${e} sourceLocA = ${e}(${l.join()}, 0);
        --${l[s-1]};
        ${e} sourceLocB = ${e}(${l.join()}, 0);
        --${l[s-2]};`}else d=s,u=`
        ${c} sourceLocR = coords;
        ++${l[s-1]};
        ${c} sourceLocG = coords;
        ++${l[s-2]};
        ${c} sourceLocA = coords;
        --${l[s-1]};
        ${c} sourceLocB = coords;
        --${l[s-2]};`;let p=[`x`,`y`,`z`,`w`,`u`,`v`].slice(0,d),m=`.`+p[d-1],h=p.map(e=>`int `+e),g=fk(`sourceLocR`,d-1).concat(`inIdx.r`),_=fk(`sourceLocG`,d-1).concat(`inIdx.g`),v=fk(`sourceLocB`,d-1).concat(`inIdx.b`),y=fk(`sourceLocA`,d-1).concat(`inIdx.a`),b=n===`max`?`greaterThan`:`lessThan`,x=r?``:`
          inIdx = round(vec4(getBestIndicesAChannel(${g.join()}),
                             getBestIndicesAChannel(${_.join()}),
                             getBestIndicesAChannel(${v.join()}),
                             getBestIndicesAChannel(${y.join()})));`,S=`vec4(
            getAChannel(${g.join()}),
            hasNextCol ? getAChannel(${_.join()}) : 0.,
            hasNextRow ? getAChannel(${v.join()}) : 0.,
            hasNextRow && hasNextCol ? getAChannel(${y.join()}) : 0.)`,C=r?``:`
      float getBestIndicesAChannel(${h.join()}) {
        return getChannel(getBestIndicesA(${p.join()}),
                                          vec2(${p.slice(-2).join()}));
      }`;this.userCode=`
      float getAChannel(${h.join()}) {
        return getChannel(getA(${p.join()}),
                               vec2(${p.slice(-2).join()}));
      }
      ${C}
      void main() {
        ${c} coords = getOutputCoords();
        bool hasNextCol = ${l[s-1]} < ${o[s-1]-1};
        bool hasNextRow = ${l[s-2]} < ${o[s-2]-1};
        ${u}
        ivec4 srcIdx = ivec4(sourceLocR${m}, sourceLocG${m},
          sourceLocB${m}, sourceLocA${m}) * ${t};
        ivec4 inIdx = srcIdx;
        vec4 bestIndex = vec4(inIdx);
        vec4 bestValue = ${S};

        for (int i = 0; i < ${t}; i++) {
          inIdx = srcIdx;
          ${x}
          vec4 candidate = ${S};
          bvec4 nan = isnan(candidate);
          bvec4 replace = bvec4(
            vec4(${b}(candidate, bestValue)) * (vec4(1.0) - vec4(nan)));

          bestValue = vec4(replace.x  ? candidate.x : bestValue.x,
                           replace.y  ? candidate.y : bestValue.y,
                           replace.z  ? candidate.z : bestValue.z,
                           replace.w  ? candidate.w : bestValue.w);
          bestIndex = mix(bestIndex, vec4(inIdx), vec4(replace));
          srcIdx++;
        }
        setOutput(bestIndex);
      }
    `}};function nj(e,t,n,r=null){let i=t.shape[0],a=t.shape[1];r!=null&&(i=r.shape[0],a=r.shape[1]);let o=S_(a),s=new ej({windowSize:o,inSize:a,batchSize:i,outSize:Math.ceil(a/o)},n,r==null),c=[t];r!=null&&c.push(r);let l=e.runWebGLProgram(s,c,`int32`);if(l.shape[1]===1)return l;let u=nj(e,t,n,l);return e.disposeIntermediateTensorInfo(l),u}function rj(e,t,n,r=null){let i=r==null?t.shape:r.shape,a=i[i.length-1],o=new tj(i,S_(a),n,r==null),s=r==null?[t]:[t,r],c=e.runWebGLProgram(o,s,`int32`);if(c.shape.length===t.shape.length){let r=rj(e,t,n,c);return e.disposeIntermediateTensorInfo(c),r}return c}function ij(e,t,n,r){let i=[n];if(gu(`arg`+r.charAt(0).toUpperCase()+r.slice(1),i,t.shape.length),!N().getBool(`WEBGL_PACK_REDUCE`)||t.shape.length<=2){let n=[],a=e.texData.get(t.dataId),o=a!==null&&a.isPacked,s=t;o&&(s=e.unpackTensor(t),n.push(s));let[c,l]=mu(s.shape,i),u=g(l),d=Q({inputs:{x:s},backend:e,attrs:{shape:[-1,u]}});n.push(d);let f=nj(e,d,r);n.push(f);let p=Q({inputs:{x:f},backend:e,attrs:{shape:c}});return n.forEach(t=>e.disposeIntermediateTensorInfo(t)),p}return rj(e,t,r)}function aj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a}=r,o=C(a,i.shape),s=_u(o,i.shape.length),c=i,l=[];s!=null&&(c=PA({inputs:{x:i},backend:n,attrs:{perm:s}}),l.push(c),o=yu(o.length,c.shape.length)),gu(`argMax`,[o[0]],c.shape.length);let u=ij(n,c,o[0],`max`);return l.forEach(e=>n.disposeIntermediateTensorInfo(e)),u}var oj={kernelName:Ae,backendName:`webgl`,kernelFunc:aj};function sj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a}=r,o=C(a,i.shape),s=_u(o,i.shape.length),c=i,l=[];s!=null&&(c=PA({inputs:{x:i},backend:n,attrs:{perm:s}}),l.push(c),o=yu(o.length,c.shape.length)),gu(`argMin`,[o[0]],c.shape.length);let u=ij(n,c,o[0],`min`);return l.forEach(e=>n.disposeIntermediateTensorInfo(e)),u}var cj={kernelName:je,backendName:`webgl`,kernelFunc:sj},lj={kernelName:Me,backendName:`webgl`,kernelFunc:Z({opSnippet:Tk+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return asin(x);
`})},uj={kernelName:Ne,backendName:`webgl`,kernelFunc:Z({opSnippet:Tk+`return log(x + sqrt(x * x + 1.0));`})},dj={kernelName:Pe,backendName:`webgl`,kernelFunc:Z({opSnippet:Tk+`
  return atan(x);
`})},fj={kernelName:Ie,backendName:`webgl`,kernelFunc:pA({opSnippet:Xk+`
  return atan(a, b);
`,packedOpSnippet:`
  vec4 result = atan(a, b);
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Qk+`
  return result;
`})},pj={kernelName:Fe,backendName:`webgl`,kernelFunc:Z({opSnippet:Tk+`
  if ((x < -1.0) || (x > 1.0)) return NAN;
return (log(1.0 + x) - log(1.0 - x)) / 2.0;`})},mj=class{constructor(e,t,n,r=!1,i=!1){if(this.variableNames=[`x`],t===`avg`&&n)throw Error(`Cannot compute positions for average pool.`);let a=e.filterWidth,o=e.strideHeight,s=e.strideWidth,c=e.dilationHeight,l=e.dilationWidth,u=e.effectiveFilterHeight,d=e.effectiveFilterWidth,f=e.padInfo.top,p=e.padInfo.left;this.outputShape=e.outShape;let m=t===`avg`,h=`((batch  * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + d`,g=`(xR * ${e.inWidth} + xC) * ${e.inChannels} + d`,_=`0.0`;if(m||(_=`-1.0 / 1e-20`),n){this.userCode=`
        const ivec2 strides = ivec2(${o}, ${s});
        const ivec2 pads = ivec2(${f}, ${p});

        void main() {
          ivec4 coords = getOutputCoords();
          int batch = coords[0];
          int d = coords[3];

          ivec2 xRCCorner = coords.yz * strides - pads;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          // max/min x(?, ?, d) to get y(yR, yC, d).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;
          float avgValue = 0.0;

          for (int wR = 0; wR < ${u};
              wR += ${c}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${d};
                wC += ${l}) {
              int xC = xCCorner + wC;

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              float value = getX(batch, xR, xC, d);

              // If a min / max value has already been found, use it. If not,
              // use the current value.
              float currMinMaxValue = mix(
                  value, minMaxValue, minMaxValueFound);
              if (value >= currMinMaxValue) {
                minMaxValue = value;
                minMaxValueFound = 1.0;
                minMaxPosition = ${r?i?h:g:`wR * ${d} + wC`};
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}let v=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t===`avg`&&(v=`avgValue / count`);let y=Math.floor(a/4)*4,b=a%4,x=`
      if (${m}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = max(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec2 strides = ivec2(${o}, ${s});
      const ivec2 pads = ivec2(${f}, ${p});
      const float initializationValue = ${_};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xR, int xC, int d) {
        if (xC < 0 || xC >= ${e.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xR, xC, d);
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // max/min x(?, ?, d) to get y(yR, yC, d).
        // ? = to be determined
        vec4 minMaxValue = vec4(${_});
        float avgValue = 0.0;
        count = 0.0;

        for (int wR = 0; wR < ${u};
            wR += ${c}) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${e.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${y}; wC += 4) {
            int xC = xCCorner + wC * ${l};

            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${l}, d),
              getValue(batch, xR, xC + 2 * ${l}, d),
              getValue(batch, xR, xC + 3 * ${l}, d)
            );

            ${x}
          }

          int xC = xCCorner + ${y};
          if (${b===1}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              initializationValue,
              initializationValue,
              initializationValue
            );

            ${x}
          } else if (${b===2}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${l}, d),
              initializationValue,
              initializationValue
            );

            ${x}
          } else if (${b===3}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${l}, d),
              getValue(batch, xR, xC + 2 * ${l}, d),
              initializationValue
            );

            ${x}
          }
        }
        setOutput(${v});
      }
    `}},hj=class{constructor(e,t,n,r=!1,i=!1){if(this.variableNames=[`x`],t===`avg`&&n)throw Error(`Cannot compute positions for average pool.`);let a=e.filterWidth,o=e.strideDepth,s=e.strideHeight,c=e.strideWidth,l=e.dilationDepth,u=e.dilationHeight,d=e.dilationWidth,f=e.effectiveFilterDepth,p=e.effectiveFilterHeight,m=e.effectiveFilterWidth,h=e.padInfo.front,g=e.padInfo.top,_=e.padInfo.left;this.outputShape=e.outShape;let v=t===`avg`,y=`0.0`;if(v||(y=`-1.0 / 1e-20`),n){this.userCode=`
        const ivec3 strides =
            ivec3(${o}, ${s}, ${c});
        const ivec3 pads = ivec3(${h}, ${g}, ${_});

        void main() {
          ivec5 coords = getOutputCoords();
          int batch = coords.x;
          int ch = coords.u;

          ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
          int xDCorner = xCorner.x;
          int xRCorner = xCorner.y;
          int xCCorner = xCorner.z;

          // max/min x(?, ?, ?, ch) to get y(yD, yR, yC, ch).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;

          for (int wD = 0; wD < ${f};
              wD += ${l}) {
            int xD = xDCorner + wD;

            if (xD < 0 || xD >= ${e.inDepth}) {
              continue;
            }

            for (int wR = 0; wR < ${p};
                wR += ${u}) {
              int xR = xRCorner + wR;

              if (xR < 0 || xR >= ${e.inHeight}) {
                continue;
              }

              for (int wC = 0; wC < ${m};
                  wC += ${d}) {
                int xC = xCCorner + wC;

                if (xC < 0 || xC >= ${e.inWidth}) {
                  continue;
                }

                float value = getX(batch, xD, xR, xC, ch);

                // If a min / max value has already been found, use it. If not,
                // use the current value.
                float currMinMaxValue = mix(
                    value, minMaxValue, minMaxValueFound);
                if (value >= currMinMaxValue) {
                  minMaxValue = value;
                  minMaxValueFound = 1.0;
                  minMaxPosition = ${r?i?`(((batch * ${e.inDepth} + xD) * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + ch`:`((xD * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + ch`:`wD * ${p} * ${m} +
                      wR * ${m} + wC`};
                }
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}let b=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t===`avg`&&(b=`avgValue / count`);let x=Math.floor(a/4)*4,S=a%4,C=`
      if (${v}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = max(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec3 strides =
        ivec3(${o}, ${s}, ${c});
      const ivec3 pads = ivec3(${h}, ${g}, ${_});
      const float initializationValue = ${y};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xD, int xR, int xC, int ch) {
        if (xC < 0 || xC >= ${e.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xD, xR, xC, ch);
      }

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xDCorner = xCorner.x;
        int xRCorner = xCorner.y;
        int xCCorner = xCorner.z;

        // max/min x(?, ?, ?, d) to get y(yD, yR, yC, ch).
        // ? = to be determined
        vec4 minMaxValue = vec4(${y});
        float avgValue = 0.0;
        count = 0.0;

        for (int wD = 0; wD < ${f};
            wD += ${l}) {
          int xD = xDCorner + wD;

          if (xD < 0 || xD >= ${e.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${p};
            wR += ${u}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${x}; wC += 4) {
              int xC = xCCorner + wC * ${d};

              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                getValue(batch, xD, xR, xC + 2 * ${d}, ch),
                getValue(batch, xD, xR, xC + 3 * ${d}, ch)
              );

              ${C}
            }

            int xC = xCCorner + ${x};
            if (${S===1}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                initializationValue,
                initializationValue,
                initializationValue
              );

              ${C}
            } else if (${S===2}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                initializationValue,
                initializationValue
              );

              ${C}
            } else if (${S===3}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                getValue(batch, xD, xR, xC + 2 * ${d}, ch),
                initializationValue
              );

              ${C}
            }
          }
          setOutput(${b});
        }
      }
    `}};function gj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t;LE(i,`avgPool`);let{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=r;f(Tc(o,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${o} and dilations '1'`);let l=dc(i.shape,a,o,1,s,c);if(l.filterWidth===1&&l.filterHeight===1&&_(l.inShape,l.outShape))return eA({inputs:{x:i},backend:n});let u=new mj(l,`avg`,!1);return n.runWebGLProgram(u,[i],`float32`)}var _j={kernelName:Le,backendName:`webgl`,kernelFunc:gj};function vj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c,dataFormat:l}=r,u=new hj(fc(i.shape,a,o,[1,1,1],s,c,l),`avg`,!1);return n.runWebGLProgram(u,[i],`float32`)}var yj={kernelName:ze,backendName:`webgl`,kernelFunc:vj},bj=class{constructor(e){this.variableNames=[`dy`],this.outputShape=e.inShape;let t=e.filterHeight,n=e.filterWidth,r=e.strideHeight,i=e.strideWidth,a=e.dilationHeight,o=e.dilationWidth,s=e.effectiveFilterHeight,c=e.effectiveFilterWidth,l=s-1-e.padInfo.top,u=c-1-e.padInfo.left,d=1/(t*n);this.userCode=`
      const ivec2 pads = ivec2(${l}, ${u});
      const float avgMultiplier = float(${d});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${s};
            wR += ${a}) {
          float dyR = float(dyRCorner + wR) / ${r}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${c};
            wC+= ${o}) {
            float dyC = float(dyCCorner + wC) / ${i}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);

            dotProd += dyValue * avgMultiplier;
          }
        }
        setOutput(dotProd);
      }
    `}},xj=class{constructor(e){this.variableNames=[`dy`],this.outputShape=e.inShape;let t=e.filterDepth,n=e.filterHeight,r=e.filterWidth,i=e.strideDepth,a=e.strideHeight,o=e.strideWidth,s=e.dilationDepth,c=e.dilationHeight,l=e.dilationWidth,u=e.effectiveFilterDepth,d=e.effectiveFilterHeight,f=e.effectiveFilterWidth,p=u-1-e.padInfo.front,m=d-1-e.padInfo.top,h=f-1-e.padInfo.left,g=1/(t*n*r);this.userCode=`
      const ivec3 pads = ivec3(${p}, ${m}, ${h});
      const float avgMultiplier = float(${g});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${u};
            wD += ${s}) {
          float dyD = float(dyDCorner + wD) / ${i}.0;

          if (dyD < 0.0 || dyD >= ${e.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${d};
              wR += ${c}) {
            float dyR = float(dyRCorner + wR) / ${a}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${f};
                wC += ${l}) {
              float dyC = float(dyCCorner + wC) / ${o}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);

              dotProd += dyValue * avgMultiplier;
            }
          }
        }
        setOutput(dotProd);
      }
    `}};function Sj(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a,{filterSize:s,strides:c,pad:l,dimRoundingMode:u}=r,d=new xj(fc(o.shape,s,c,[1,1,1],l,u));return n.runWebGLProgram(d,[i],o.dtype)}var Cj={kernelName:Be,backendName:`webgl`,kernelFunc:Sj};function wj(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a;LE([i,a],`avgPoolGrad`);let{filterSize:s,strides:c,pad:l}=r,u=new bj(dc(o.shape,s,c,1,l));return n.runWebGLProgram(u,[i],o.dtype)}var Tj={kernelName:Re,backendName:`webgl`,kernelFunc:wj};function Ej(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a}=t,{transposeA:o,transposeB:s}=r;return IA({a:i,b:a,transposeA:o,transposeB:s,backend:n})}var Dj={kernelName:Ve,backendName:`webgl`,kernelFunc:Ej},Oj=class{constructor(e,t,n,r,i,a){this.outputShape=[],this.variableNames=[`x`,`mean`,`variance`],z(e,t),z(e,n);let o=`0.0`;r!=null&&(z(e,r),this.variableNames.push(`offset`),o=`getOffsetAtOutCoords()`);let s=`1.0`;i!=null&&(z(e,i),this.variableNames.push(`scale`),s=`getScaleAtOutCoords()`),this.outputShape=e,this.userCode=`
      void main() {
        float x = getXAtOutCoords();
        float mean = getMeanAtOutCoords();
        float variance = getVarianceAtOutCoords();
        float offset = ${o};
        float scale = ${s};
        float inv = scale * inversesqrt(variance + float(${a}));
        setOutput(dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));
      }
    `}},kj=class{constructor(e,t,n,r,i,a){this.packedInputs=!0,this.packedOutput=!0,this.variableNames=[`x`,`mean`,`variance`],z(e,t),z(e,n);let o=`vec4(0.0)`;r!=null&&(z(e,r),this.variableNames.push(`offset`),o=`getOffsetAtOutCoords()`);let s=`vec4(1.0)`;i!=null&&(z(e,i),this.variableNames.push(`scale`),s=`getScaleAtOutCoords()`),this.outputShape=e,this.userCode=`
      void main() {
        vec4 offset = ${o};
        vec4 scale = ${s};

        vec4 x = getXAtOutCoords();
        vec4 mean = getMeanAtOutCoords();
        vec4 variance = getVarianceAtOutCoords();

        vec4 inv = scale * inversesqrt(variance + vec4(${a}));

        setOutput((x - mean) * inv + offset);
      }
    `}},Aj={kernelName:Et,backendName:`webgl`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{let{x:r,mean:i,variance:a,offset:o,scale:s}=e;f(i.shape.length===a.shape.length,()=>`Batch normalization gradient requires mean and variance to have equal ranks.`),f(o==null||i.shape.length===o.shape.length,()=>`Batch normalization gradient requires mean and offset to have equal ranks.`),f(s==null||i.shape.length===s.shape.length,()=>`Batch normalization gradient requires mean and scale to have equal ranks.`);let{varianceEpsilon:c}=n;c??=.001;let l=[r,i,a],u=null;o!=null&&(u=o.shape,l.push(o));let d=null;s!=null&&(d=s.shape,l.push(s));let p=N().getBool(`WEBGL_PACK_NORMALIZATION`)?new kj(r.shape,i.shape,a.shape,u,d,c):new Oj(r.shape,i.shape,a.shape,u,d,c);return t.runWebGLProgram(p,l,l[0].dtype)}},jj=class{constructor(e){this.variableNames=[`source`],this.outputShape=e,this.rank=e.length;let t=X(this.rank);this.customUniforms=[{name:`start`,arrayIndex:this.rank,type:`int`}];let n=Nj(this.rank),r;r=`
        ${t} sourceLoc;
        ${t} coords = getOutputCoords();
        ${e.map((e,t)=>`sourceLoc.${Mj[t]} = start[${t}] + coords.${Mj[t]};`).join(`
`)}
      `,this.userCode=`
      void main() {
        ${r}
        setOutput(getSource(${n}));
      }
    `}},Mj=[`x`,`y`,`z`,`w`,`u`,`v`];function Nj(e){if(e===1)return`sourceLoc`;if(e<=6)return Mj.slice(0,e).map(e=>`sourceLoc.`+e).join(`,`);throw Error(`Slicing for rank ${e} is not yet supported`)}var Pj=class{constructor(e){this.variableNames=[`source`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.rank=e.length,this.customUniforms=[{name:`start`,arrayIndex:this.rank,type:`int`}];let t=X(this.rank),n=fk(`coords`,this.rank),r=fk(`sourceLoc`,this.rank),i=this.rank===1?`sourceLoc`:`vec2(${r.slice(-2).join()})`,a=`getChannel(getSource(${r.join()}), ${i})`,o=`
      result.x = ${a};
      if (++${n[this.rank-1]} < ${e[this.rank-1]}) {
        ++${r[this.rank-1]};
        result.y = ${a};
        --${r[this.rank-1]};
      }
    `,s=this.rank===1?``:`
      --${n[this.rank-1]};
      if (++${n[this.rank-2]} < ${e[this.rank-2]}) {
        ++${r[this.rank-2]};
        result.z = ${a};
        if (++${n[this.rank-1]} < ${e[this.rank-1]}) {
          ++${r[this.rank-1]};
          result.w = ${a};
        }
      }
    `,c=this.rank<=4?`sourceLoc = coords +
            ${t}(${e.map((e,t)=>`start[${t}]`).join()});`:e.map((e,t)=>`${r[t]} = ${n[t]} + start[${t}];`).join(`
`);this.userCode=`
      void main() {
        ${t} coords = getOutputCoords();
        ${t} sourceLoc;
        ${c}
        vec4 result = vec4(0.);
        ${o}
        ${s}
        setOutput(result);
      }
    `}};function Fj(e,t,n,r){let i=r.texData.get(e.dataId),a=r.makeTensorInfo(n,e.dtype),o=r.texData.get(a.dataId);Object.assign(o,i),o.refCount=1,o.shape=n,o.dtype=e.dtype;let s=Es(t,j(e.shape));i.slice&&(s+=i.slice.flatOffset),o.slice={flatOffset:s,origDataId:i.slice&&i.slice.origDataId||e.dataId};let c=r.dataRefCount.get(o.slice.origDataId)||1;return r.dataRefCount.set(o.slice.origDataId,c+1),a}function Ij(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{begin:a,size:o}=r,[s,c]=Ds(i,a,o);if(ps(i,s,c),g(c)===0)return n.makeTensorInfo(c,i.dtype,[]);if(n.shouldExecuteOnCPU([i])||i.dtype===`string`){let e=ZO(n.texData.get(i.dataId).values,s,c,i.shape,i.dtype);return n.makeTensorInfo(c,i.dtype,e)}let{isPacked:l}=n.texData.get(i.dataId),u=Ts(i.shape,s,c);if(l||!u){let e=N().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new Pj(c):new jj(c),t=[s];return n.runWebGLProgram(e,[i],i.dtype,t)}return n.uploadToGPU(i.dataId),Fj(i,s,c,n)}var Lj={kernelName:Nn,backendName:`webgl`,kernelFunc:Ij},Rj={kernelName:He,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,crops:o}=r;f(i.shape.length<=4,()=>`batchToSpaceND for rank > 4 with a WebGL backend not implemented yet`);let s=a.reduce((e,t)=>e*t),c=w_(i.shape,a,s),l=T_(c.length,a.length),u=E_(i.shape,a,s),d=D_(o,a.length),p=O_(u,o,a.length),m=[],h=Q({inputs:{x:i},backend:n,attrs:{shape:c}}),g=PA({inputs:{x:h},backend:n,attrs:{perm:l}}),_=Q({inputs:{x:g},backend:n,attrs:{shape:u}}),v=Ij({inputs:{x:_},backend:n,attrs:{begin:d,size:p}});return m.push(h),m.push(g),m.push(_),m.forEach(e=>n.disposeIntermediateTensorInfo(e)),v}};function zj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,weights:a}=t,{size:o}=r,s=bO(n.readSync(i.dataId),n.readSync(a.dataId),a.dtype,a.shape,o);return n.makeTensorInfo([o],a.dtype,s)}var Bj={kernelName:Ue,backendName:`webgl`,kernelFunc:zj};function Vj(e){let{inputs:t,backend:n}=e,{s0:r,s1:i}=t,a=n.readSync(r.dataId),o=n.readSync(i.dataId),s=z(Array.from(a),Array.from(o));return n.makeTensorInfo([s.length],`int32`,Int32Array.from(s))}var Hj={kernelName:We,backendName:`webgl`,kernelFunc:Vj},Uj=pA({opSnippet:`return float(a != b);`,cpuKernelImpl:HO,dtype:`bool`}),Wj={kernelName:rn,backendName:`webgl`,kernelFunc:Uj};function Gj(e){let{inputs:t,backend:n}=e,{input:r}=t;return eA({inputs:{x:n.texData.get(r.dataId).complexTensorInfos.real},backend:n})}var Kj={kernelName:_n,backendName:`webgl`,kernelFunc:Gj},qj=`return float(int(x));`;function Jj(e,t){let n=new wk(e.shape,qj),r=t.runWebGLProgram(n,[e],`int32`);return{dataId:r.dataId,shape:r.shape,dtype:r.dtype}}function Yj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dtype:a}=r;if(a===`complex64`){if(i.dtype===`complex64`)return eA({inputs:{x:i},backend:n});let e=Qd(i.shape),t=Yj({inputs:{x:i},backend:n,attrs:{dtype:`float32`}}),r=nA({inputs:{real:t,imag:e},backend:n});return e.dispose(),n.disposeIntermediateTensorInfo(t),r}if(i.dtype===`complex64`){let e=Gj({inputs:{input:i},backend:n}),t=Yj({inputs:{x:e},backend:n,attrs:{dtype:a}});return n.disposeIntermediateTensorInfo(e),t}if(!ee(i.dtype,a)){let e=eA({inputs:{x:i},backend:n});return{dataId:e.dataId,shape:e.shape,dtype:a}}if(n.shouldExecuteOnCPU([i])){let e=n.texData.get(i.dataId).values,[t,r,o]=SO(e,i.shape,i.dtype,a);return n.makeTensorInfo(t,r,o)}if(a===`int32`)return Jj(i,n);if(a===`bool`){let e=n.makeTensorInfo([],`bool`,T(`bool`,1)),t=Uj({inputs:{a:i,b:e},backend:n});return n.disposeIntermediateTensorInfo(e),t}throw Error(`Error in Cast: failed to cast ${i.dtype} to ${a}`)}var Xj={kernelName:Ge,backendName:`webgl`,kernelFunc:Yj},Zj=`return ceil(x);`,Qj={kernelName:Ke,backendName:`webgl`,kernelFunc:Z({opSnippet:Zj,packedOpSnippet:Zj,cpuKernelImpl:CO})},$j=class{constructor(e){this.variableNames=[`A`],this.customUniforms=[{name:`minVal`,type:`float`},{name:`maxVal`,type:`float`}],this.outputShape=e,this.userCode=`

      void main() {
        float value = getAAtOutCoords();
        if (isnan(value)) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, minVal, maxVal));
      }
    `}},eM=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`minVal`,type:`float`},{name:`maxVal`,type:`float`}],this.outputShape=e,this.userCode=`
      void main() {
        vec4 value = getAAtOutCoords();

        if (any(isnan(value))) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, vec4(minVal), vec4(maxVal)));
      }
    `}};function tM(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{clipValueMin:a,clipValueMax:o}=r,s;s=N().getBool(`WEBGL_PACK_CLIP`)?new eM(i.shape):new $j(i.shape);let c=[[a],[o]];return n.runWebGLProgram(s,[i],i.dtype,c)}var nM={kernelName:qe,backendName:`webgl`,kernelFunc:tM},rM=class{constructor(e){this.variableNames=[`real`,`imag`],this.outputShape=e,this.userCode=`
      void main() {
        float re = abs(getRealAtOutCoords());
        float im = abs(getImagAtOutCoords());
        float mx = max(re, im);

        // sadly the length function in glsl is not underflow-safe
        // (at least not on Intel GPUs). So the safe solution is
        // to ensure underflow-safety in all cases.
        setOutput(
          mx == 0.0 ? 0.0 : mx * length(vec2(1, min(re, im)/mx))
        );
      }
    `}};function iM(e,t){return{dataId:t.dataId,dtype:t.dtype,shape:e.shape}}function aM(e){let{inputs:t,backend:n}=e,{x:r}=t,i=n.texData.get(r.dataId),a=new rM(r.shape),o=[iM(r,i.complexTensorInfos.real),iM(r,i.complexTensorInfos.imag)];return n.runWebGLProgram(a,o,o[0].dtype)}var oM={kernelName:Ye,backendName:`webgl`,kernelFunc:aM},sM=class{constructor(e){this.outputShape=[],this.outputShape=g_(e,1),this.variableNames=e.map((e,t)=>`T${t}`);let t=Array(e.length-1);t[0]=e[0][1];for(let n=1;n<t.length;n++)t[n]=t[n-1]+e[n][1];let n=[`if (yC < ${t[0]}) setOutput(getT0(yR, yC));`];for(let e=1;e<t.length;e++){let r=t[e-1];n.push(`else if (yC < ${t[e]}) setOutput(getT${e}(yR, yC-${r}));`)}let r=t.length,i=t[t.length-1];n.push(`else setOutput(getT${r}(yR, yC-${i}));`),this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int yR = coords.x;
        int yC = coords.y;

        ${n.join(`
        `)}
      }
    `}},cM=class{constructor(e,t){this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[],this.outputShape=g_(e,t);let n=this.outputShape,r=n.length,i=X(r),a=fk(`coords`,r),o=[`x`,`y`,`z`,`w`,`u`,`v`].slice(0,r);this.variableNames=e.map((e,t)=>`T${t}`);let s=Array(e.length-1);s[0]=e[0][t];for(let n=1;n<s.length;n++)s[n]=s[n-1]+e[n][t];let c=o[t],l=o.slice(-2),u=o.join(),d=`if (${c} < ${s[0]}) {
        return getChannel(
            getT0(${u}), vec2(${l.join()}));
        }`;for(let e=1;e<s.length;e++){let t=s[e-1];d+=`
        if (${c} < ${s[e]}  && ${c} >= ${s[e-1]}) {
          return getChannel(
            getT${e}(${lM(o,c,t)}),
            vec2(${lM(l,c,t)}));
        }`}let f=s.length,p=s[s.length-1];d+=`
        return getChannel(
          getT${f}(${lM(o,c,p)}),
          vec2(${lM(l,c,p)}));`,this.userCode=`
      float getValue(${o.map(e=>`int `+e)}) {
        ${d}
      }

      void main() {
        ${i} coords = getOutputCoords();
        vec4 result = vec4(getValue(${a}), 0., 0., 0.);

        ${a[r-1]} = ${a[r-1]} + 1;
        if (${a[r-1]} < ${n[r-1]}) {
          result.g = getValue(${a});
        }

        ${a[r-2]} = ${a[r-2]} + 1;
        if (${a[r-2]} < ${n[r-2]}) {
          result.a = getValue(${a});
        }

        ${a[r-1]} = ${a[r-1]} - 1;
        if (${a[r-2]} < ${n[r-2]} &&
            ${a[r-1]} < ${n[r-1]}) {
          result.b = getValue(${a});
        }
        setOutput(result);
      }
    `}};function lM(e,t,n){let r=e.indexOf(t);return e.map((e,t)=>t===r?`${e} - ${n}`:e).join()}function uM(e){let{inputs:t,backend:n}=e,{input:r}=t;return eA({inputs:{x:n.texData.get(r.dataId).complexTensorInfos.imag},backend:n})}var dM={kernelName:Nt,backendName:`webgl`,kernelFunc:uM};function fM(e,t,n){let r=e[0].dtype;if(r===`complex64`){let r=e.map(e=>Gj({inputs:{input:e},backend:n})),i=e.map(e=>uM({inputs:{input:e},backend:n})),a=fM(r,t,n),o=fM(i,t,n),s=nA({inputs:{real:a,imag:o},backend:n});return r.forEach(e=>n.disposeIntermediateTensorInfo(e)),i.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.disposeIntermediateTensorInfo(a),n.disposeIntermediateTensorInfo(o),s}let i=n.shouldExecuteOnCPU(e);if(r===`string`&&(i=!0),i){let i=e.map(e=>{let r=[-1,g(e.shape.slice(t))];return Q({inputs:{x:e},backend:n,attrs:{shape:r}})}),a=wO(i.map(e=>({vals:n.readSync(e.dataId),shape:e.shape})),g_(i.map(e=>e.shape),1),r,i[0].shape[0]===1),o=g_(e.map(e=>e.shape),t),s=n.makeTensorInfo(o,r,a);return i.forEach(e=>n.disposeIntermediateTensorInfo(e)),s}let a=N().getNumber(`WEBGL_MAX_TEXTURES_IN_SHADER`);if(e.length>a){let r=[];for(let i=0;i<e.length;i+=a){let o=e.slice(i,i+a);r.push(fM(o,t,n))}let i=fM(r,t,n);for(let e of r)n.disposeIntermediateTensorInfo(e);return i}if(N().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)&&e[0].shape.length>1){let i=new cM(e.map(e=>e.shape),t);return n.runWebGLProgram(i,e,r)}let{tensors2D:o,outShape:s}=pM(e,t,n),c=new sM(o.map(e=>e.shape)),l=n.runWebGLProgram(c,o,r);o.forEach(e=>n.disposeIntermediateTensorInfo(e));let u=Q({inputs:{x:l},attrs:{shape:s},backend:n});return n.disposeIntermediateTensorInfo(l),u}function pM(e,t,n){let r=g_(e.map(e=>e.shape),t);return{tensors2D:e.map(e=>Q({inputs:{x:e},attrs:{shape:[-1,g(e.shape.slice(t))]},backend:n})),outShape:r}}function mM(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r,a=C(i,t[0].shape)[0];h_(t.map(e=>e.shape),a);let o=g_(t.map(e=>e.shape),a);if(g(o)===0)return n.makeTensorInfo(o,t[0].dtype,[]);let s=t.filter(e=>g(e.shape)>0);return s.length===1?eA({inputs:{x:s[0]},backend:n}):fM(s,a,n)}var hM={kernelName:Xe,backendName:`webgl`,kernelFunc:mM},gM=class{constructor(e,t=!1,n=null,r=!1,i=!1){this.variableNames=[`x`,`W`],this.outputShape=e.outShape;let a=e.padInfo.top,o=e.padInfo.left,s=e.strideHeight,c=e.strideWidth,l=e.dilationHeight,u=e.dilationWidth,d=e.filterHeight,f=e.filterWidth,p=Math.floor(e.inChannels/4)*4,m=e.inChannels%4,h=e.dataFormat===`channelsLast`,g=h?1:2,_=h?2:3,v=h?3:1,y=``,b=``;n&&(y=r?`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${n}
        }`:i?`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${n}
        }`:`
          float activation(float x) {
            ${n}
          }
        `,b=`result = activation(result);`);let x=t?`result += getBiasAtOutCoords();`:``;t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),i&&this.variableNames.push(`leakyreluAlpha`),this.userCode=`
      ${y}

      const ivec2 strides = ivec2(${s}, ${c});
      const ivec2 pads = ivec2(${a}, ${o});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d2 = coords[${v}];

        ivec2 xRCCorner =
            ivec2(coords[${g}], coords[${_}]) * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${d}; wR++) {
          int xR = xRCorner + wR * ${l};

          if (xR < 0 || xR >= ${e.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${f}; wC++) {
            int xC = xCCorner + wC * ${u};

            if (xC < 0 || xC >= ${e.inWidth}) {
              continue;
            }

            for (int d1 = 0; d1 < ${p}; d1 += 4) {
              vec4 wValues = vec4(
                getW(wR, wC, d1, d2),
                getW(wR, wC, d1 + 1, d2),
                getW(wR, wC, d1 + 2, d2),
                getW(wR, wC, d1 + 3, d2)
              );

              if (${h}) {
                vec4 xValues = vec4(
                  getX(batch, xR, xC, d1),
                  getX(batch, xR, xC, d1 + 1),
                  getX(batch, xR, xC, d1 + 2),
                  getX(batch, xR, xC, d1 + 3)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec4 xValues = vec4(
                  getX(batch, d1, xR, xC),
                  getX(batch, d1 + 1, xR, xC),
                  getX(batch, d1 + 2, xR, xC),
                  getX(batch, d1 + 3, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }
            }

            if (${m===1}) {

              if (${h}) {
                dotProd +=
                    getX(batch, xR, xC, ${p}) *
                    getW(wR, wC, ${p}, d2);
              } else {
                dotProd +=
                    getX(batch, ${p}, xR, xC) *
                    getW(wR, wC, ${p}, d2);
              }

            } else if (${m===2}) {
              vec2 wValues = vec2(
                getW(wR, wC, ${p}, d2),
                getW(wR, wC, ${p} + 1, d2)
              );

              if (${h}) {
                vec2 xValues = vec2(
                  getX(batch, xR, xC, ${p}),
                  getX(batch, xR, xC, ${p} + 1)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec2 xValues = vec2(
                  getX(batch, ${p}, xR, xC),
                  getX(batch, ${p} + 1, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            } else if (${m===3}) {
              vec3 wValues = vec3(
                getW(wR, wC, ${p}, d2),
                getW(wR, wC, ${p} + 1, d2),
                getW(wR, wC, ${p} + 2, d2)
              );

              if (${h}) {
                vec3 xValues = vec3(
                  getX(batch, xR, xC, ${p}),
                  getX(batch, xR, xC, ${p} + 1),
                  getX(batch, xR, xC, ${p} + 2)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec3 xValues = vec3(
                  getX(batch, ${p}, xR, xC),
                  getX(batch, ${p} + 1, xR, xC),
                  getX(batch, ${p} + 2, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            }
          }
        }

        float result = dotProd;
        ${x}
        ${b}
        setOutput(result);
      }
    `}},_M=class{constructor(e){this.variableNames=[`x`,`W`],this.outputShape=e.outShape;let t=e.padInfo.front,n=e.padInfo.top,r=e.padInfo.left,i=e.strideDepth,a=e.strideHeight,o=e.strideWidth,s=e.dilationDepth,c=e.dilationHeight,l=e.dilationWidth,u=e.filterDepth,d=e.filterHeight,f=e.filterWidth,p=Math.floor(e.inChannels/4)*4,m=e.inChannels%4;this.userCode=`
      const ivec3 strides = ivec3(${i}, ${a}, ${o});
      const ivec3 pads = ivec3(${t}, ${n}, ${r});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d2 = coords.u;

        ivec3 xFRCCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xFCorner = xFRCCorner.x;
        int xRCorner = xFRCCorner.y;
        int xCCorner = xFRCCorner.z;

        // Convolve x(?, ?, ?, d1) with w(:, :, :, d1, d2) to get
        // y(yF, yR, yC, d2). ? = to be determined. : = across all
        // values in that axis.
        float dotProd = 0.0;
        for (int wF = 0; wF < ${u}; wF++) {
          int xF = xFCorner + wF * ${s};

          if (xF < 0 || xF >= ${e.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${d}; wR++) {
            int xR = xRCorner + wR * ${c};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${f}; wC++) {
              int xC = xCCorner + wC * ${l};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              for (int d1 = 0; d1 < ${p}; d1 += 4) {
                vec4 xValues = vec4(
                  getX(batch, xF, xR, xC, d1),
                  getX(batch, xF, xR, xC, d1 + 1),
                  getX(batch, xF, xR, xC, d1 + 2),
                  getX(batch, xF, xR, xC, d1 + 3)
                );
                vec4 wValues = vec4(
                  getW(wF, wR, wC, d1, d2),
                  getW(wF, wR, wC, d1 + 1, d2),
                  getW(wF, wR, wC, d1 + 2, d2),
                  getW(wF, wR, wC, d1 + 3, d2)
                );

                dotProd += dot(xValues, wValues);
              }

              if (${m===1}) {
                dotProd +=
                  getX(batch, xF, xR, xC, ${p}) *
                  getW(wF, wR, wC, ${p}, d2);
              } else if (${m===2}) {
                vec2 xValues = vec2(
                  getX(batch, xF, xR, xC, ${p}),
                  getX(batch, xF, xR, xC, ${p} + 1)
                );
                vec2 wValues = vec2(
                  getW(wF, wR, wC, ${p}, d2),
                  getW(wF, wR, wC, ${p} + 1, d2)
                );
                dotProd += dot(xValues, wValues);
              } else if (${m===3}) {
                vec3 xValues = vec3(
                  getX(batch, xF, xR, xC, ${p}),
                  getX(batch, xF, xR, xC, ${p} + 1),
                  getX(batch, xF, xR, xC, ${p} + 2)
                );
                vec3 wValues = vec3(
                  getW(wF, wR, wC, ${p}, d2),
                  getW(wF, wR, wC, ${p} + 1, d2),
                  getW(wF, wR, wC, ${p} + 2, d2)
                );
                dotProd += dot(xValues, wValues);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}},vM=class{constructor(e,t=!1,n=null,r=!1,i=!1){this.variableNames=[`x`,`W`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`pads`,type:`ivec2`},{name:`strides`,type:`ivec2`},{name:`dilations`,type:`ivec2`},{name:`inDims`,type:`ivec2`}],this.outputShape=e.outShape,this.enableShapeUniforms=HD(this.outputShape.length);let a=e.padInfo.left,o=e.strideWidth,s=e.dilationWidth,c=e.filterHeight,u=e.filterWidth,d=u,f=`
       int xR; int xC; int xCOffset;
       vec4 wTexel; vec4 previous; vec4 final;`;for(let e=0;e<u;e++)f+=`
           vec4 xTexelC${e*2};
           int xTexelC${e*2}Ready;
           vec4 xTexelC${e*2+1};
           int xTexelC${e*2+1}Ready;
           vec4 xC${e};`;f+=`
     for (int r = 0; r < ${c}; r++) {
      for (int d1 = 0; d1 < ${e.inChannels}; d1 += 2) {
       `;for(let e=0;e<u;e++)f+=`
           xTexelC${e*2} = vec4(0.0);
           xTexelC${e*2}Ready = 0;
           xTexelC${e*2+1} = vec4(0.0);
           xTexelC${e*2+1}Ready = 0;
           xC${e} = vec4(0.0);`;f+=`
         xR = xRCorner + r * dilations[0];
         if (xR >=0 && xR < inDims[0]) {
       `;for(let t=0;t<(d+1)/2;t++){let n=t*2;if(f+=`
           xC = xCCorner + ${n*s};
           `,o===1){if(n<u&&(a%2==1?(f+=`
                 xCOffset = xC + 1;
                 if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n}Ready == 0) {
                   xTexelC${n} = getX(batch, xR, xCOffset, d1);

                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${n}.zw = vec2(0.0);
                   }
                   xTexelC${n}Ready = 1;
                 }
               `,f+=s===1&&n>0?`
                 xC${n} = vec4(xTexelC${n-2}.zw, xTexelC${n}.xy);
                 `:`
                   xCOffset = xC + 1 - 2;

                   if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       previous.zw = vec2(0.0);
                     }

                     xC${n} = vec4(previous.zw, xTexelC${n}.xy);
                   } else {
                     xC${n} = vec4(0.0, 0.0, xTexelC${n}.xy);
                   }
                   `):f+=`
                 if (xC >= 0 && xC < inDims[1] && xTexelC${n}Ready == 0) {
                   xTexelC${n} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${n}.zw = vec2(0.0);
                   }
                   xTexelC${n}Ready = 1;
                 }

                 xC${n} = xTexelC${n};
                 `,n+1<u)){let e=a%2==0?l(s):s;s%2==0&&a%2==1||s%2!=0&&a%2!=1?(f+=`
                   xCOffset = xC + imod(pads[1], 2) + ${e};

                   if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n+1}Ready == 0) {
                     xTexelC${n+1} = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       xTexelC${n+1}.zw = vec2(0.0);
                     }
                     xTexelC${n+1}Ready = 1;
                   }
                   `,f+=s>1?`
                     xCOffset -= 2;
                     if (xCOffset >= 0 && xCOffset < inDims[1]) {
                      previous = getX(batch, xR, xCOffset, d1);
                      xC${n+1} = vec4(previous.zw, xTexelC${n+1}.xy);
                     } else {
                      xC${n+1} = vec4(0.0, 0.0, xTexelC${n+1}.xy);
                     }
                     `:`
                     xC${n+1} = vec4(xTexelC${n}.zw, xTexelC${n+1}.xy);
                     `):f+=e===1?`
                     xC${n+1} = xTexelC${n};
                     `:`
                     xCOffset = xC + ${e};

                     if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n+1}Ready == 0) {
                       xTexelC${n+1} = getX(batch, xR, xCOffset, d1);
                       if (xCOffset + 1 >= inDims[1]) {
                         xTexelC${n+1}.zw = vec2(0.0);
                       }
                       xTexelC${n+1}Ready = 1;
                     }

                     xC${n+1} = xTexelC${n+1};
                     `}}else n<u&&(a%2==1?(f+=`
                 xCOffset = xC + 1 - strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n}Ready == 0) {
                   xTexelC${n} = getX(batch, xR, xCOffset, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${n}.zw = vec2(0.0);
                   }
                   xTexelC${n}Ready = 1;
                 }

                 if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${n+1}Ready == 0) {
                   xTexelC${n+1} = getX(batch, xR, xC + 1, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xC + 2 >= inDims[1]) {
                     xTexelC${n+1}.zw = vec2(0.0);
                   }
                   xTexelC${n+1}Ready = 1;
                 }

                 xC${n} = vec4(xTexelC${n}.zw, xTexelC${n+1}.zw);
               `,n+1<u&&(f+=`
                   final = vec4(0.0);
                   xCOffset = xC + 1 + strides[1];
                   if(xCOffset >= 0 && xCOffset < inDims[1]) {
                     final = getX(batch, xR, xCOffset, d1);
                   }
                   xC${n+1} = vec4(xTexelC${n+1}.xy, final.xy);
                 `)):(f+=`
                 if(xC >= 0 && xC < inDims[1] && xTexelC${n}Ready == 0) {
                   xTexelC${n} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${n}.zw = vec2(0.0);
                   }
                   xTexelC${n}Ready = 1;
                 }

                 xCOffset = xC + strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n+1}Ready == 0) {
                   xTexelC${n+1} = getX(batch, xR, xCOffset, d1);
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${n+1}.zw = vec2(0.);
                   }
                   xTexelC${n+1}Ready = 1;
                 }

                 xC${n} = vec4(
                   xTexelC${n}.xy, xTexelC${n+1}.xy);
               `,n+1<u&&(f+=`
                   xC${n+1} = vec4(xTexelC${n}.zw, xTexelC${n+1}.zw);
                 `)));n<u&&(f+=`
             wTexel = getW(r, ${n}, d1, d2);
             dotProd += xC${n}.xxzz * vec4(wTexel.xy, wTexel.xy);
             if(d1 + 1 < ${e.inChannels}) {
               dotProd += xC${n}.yyww * vec4(wTexel.zw, wTexel.zw);
             }
           `,n+1<u&&(f+=`
               wTexel = getW(r, ${n+1}, d1, d2);
               dotProd += xC${n+1}.xxzz * vec4(wTexel.xy, wTexel.xy);
               if(d1 + 1 < ${e.inChannels}) {
                 dotProd += xC${n+1}.yyww * vec4(wTexel.zw, wTexel.zw);
               }
             `))}f+=`
     }
   `,f+=`
     }
   `,f+=`
     }
   `;let p=``,m=``;n&&(p=r?`vec4 activation(vec4 a) {
           vec4 b = getPreluActivationWeightsAtOutCoords();
           ${n}
         }`:i?`vec4 activation(vec4 a) {
           vec4 b = getLeakyreluAlphaAtOutCoords();
           ${n}
         }`:`vec4 activation(vec4 x) {
           ${n}
         }`,m=`result = activation(result);`);let h=t?`result += getBiasAtOutCoords();`:``;t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),i&&this.variableNames.push(`leakyreluAlpha`),this.userCode=`
       ${p}

       void main() {
         ivec4 coords = getOutputCoords();
         int batch = coords.x;
         ivec2 xRCCorner = coords.yz * strides - pads;
         int d2 = coords.w;
         int xRCorner = xRCCorner.x;
         int xCCorner = xRCCorner.y;

         //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
         vec4 dotProd = vec4(0.000000000000001);

         ${f}

         vec4 result = dotProd - vec4(0.000000000000001);
         ${h}
         ${m}
         setOutput(result);
       }
     `}},yM=class{constructor(e,t){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`inputShape`,type:`ivec4`},{name:`pad`,type:`ivec2`},{name:`stride`,type:`ivec2`},{name:`dilation`,type:`ivec2`},{name:`inChannels`,type:`int`},{name:`itemsPerBlockRow`,type:`int`},{name:`outWidth`,type:`int`}],this.outputShape=e,this.enableShapeUniforms=HD(this.outputShape.length);let{dataFormat:n}=t,r=RE(),i=n===`channelsLast`,a=i?1:2,o=i?2:3,s=this.enableShapeUniforms?`if(blockIndex < outShape[2] && pos < outShape[1]) {`:`if(blockIndex < ${e[2]} && pos < ${e[1]}) {`,c=``;for(let e=0;e<=1;e++)for(let t=0;t<=1;t++)c+=`
          blockIndex = rc.z + ${t};
          pos = rc.y + ${e};

          ${s}
            offsetY = int(blockIndex / outWidth) * stride[0] - pad[0];
            d0 = offsetY + dilation[0] * (pos / itemsPerBlockRow);

            if(d0 < inputShape[${a}] && d0 >= 0) {
              // Use custom imod instead mod. On Intel GPU, mod may generate
              // unexpected value.
              // https://github.com/tensorflow/tfjs/issues/5447
              offsetX = imod(blockIndex, outWidth) * stride[1] - pad[1];
              d1 = offsetX + dilation[1] * (imod(pos, itemsPerBlockRow) /
                  inChannels);

              if(d1 < inputShape[${o}] && d1 >= 0) {

                ch = imod(pos, inChannels);

                if (${i}) {
                  innerDims = vec2(d1, ch);
                  result[${e*2+t}] = getChannel(
                    getA(rc.x, d0, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                } else {
                  innerDims = vec2(d0, d1);
                  result[${e*2+t}] = getChannel(
                    getA(rc.x, ch, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                }
              }
            }
          }
        `;this.userCode=`
      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0);

        int blockIndex, pos, offsetY, d0, offsetX, d1, ch;
        vec2 innerDims;

        ${c}

        ${r.output} = result;
      }
    `}};function bM(e,t){let n=e.length;return n>=3?t?[...e.slice(0,-3),e[n-3]*e[n-2],e[n-1]]:[...e.slice(0,-3),e[n-3],e[n-2]*e[n-1]]:!t&&n===1&&e[0]>1?[e[0],1]:null}function xM({x:e,filter:t,convInfo:n,backend:r,bias:i=null,preluActivationWeights:a=null,leakyreluAlpha:o=0,activation:s=null}){let c=e.shape,l=r.texData.get(e.dataId),u=n.inChannels,d=c[0]*c[1]*c[2],p=n.outChannels,m=n.dataFormat===`channelsLast`,h,g=[];if(a!=null){let e=bM(a.shape,m);e!=null&&(a=Q({inputs:{x:a},backend:r,attrs:{shape:e}}),g.push(a))}if(i!=null){let e=bM(i.shape,m);e!=null&&(i=Q({inputs:{x:i},backend:r,attrs:{shape:e}}),g.push(i))}if(!((d===1||p===1)&&u>1e3)&&l.isPacked&&m&&l.texture!=null&&c[2]%2!=0&&_(l.shape.slice(-3),c.slice(-3))){let u=c[0]*c[1]*(c[2]+1),d={dataId:e.dataId,shape:[1,u,n.inChannels],dtype:e.dtype},p=l.shape;l.shape=l.shape.slice(),l.shape[l.shape.length-2]++,f(wE(l.shape,d.shape),()=>`packed reshape ${l.shape} to ${d.shape} isn't free`);let m=Q({inputs:{x:t},backend:r,attrs:{shape:[1,n.inChannels,n.outChannels]}});g.push(m);let _=IA({a:d,b:m,backend:r,transposeA:!1,transposeB:!1,bias:i,activation:s,preluActivationWeights:a,leakyreluAlpha:o}),v=r.texData.get(_.dataId);f(v.isPacked,()=>`batchMatMul result is expected to be packed`),l.shape=p,v.shape=n.outShape,h=eA({inputs:{x:_},backend:r}),h.shape=n.outShape,g.push(_)}else{let c=n.outHeight*n.outWidth,l=Q({inputs:{x:e},backend:r,attrs:{shape:m?[n.batchSize,c,n.inChannels]:[n.batchSize,n.inChannels,c]}}),u=Q({inputs:{x:t},backend:r,attrs:{shape:[1,n.inChannels,n.outChannels]}}),d=IA({a:m?l:u,b:m?u:l,transposeA:!m,transposeB:!1,backend:r,bias:i,activation:s,preluActivationWeights:a,leakyreluAlpha:o});h=Q({inputs:{x:d},backend:r,attrs:{shape:n.outShape}}),g.push(l),g.push(u),g.push(d)}for(let e of g)r.disposeIntermediateTensorInfo(e);return h}function SM({x:e,filter:t,convInfo:n,backend:r,bias:i=null,preluActivationWeights:a=null,leakyreluAlpha:o=0,activation:s=null}){let{filterWidth:c,filterHeight:l,inChannels:u,outWidth:d,outHeight:f,dataFormat:p}=n,m=p===`channelsLast`,h=c*l*u,_=f*d,v=[n.batchSize,h,_],y=[];if(a!=null){let e=bM(a.shape,m);e!=null&&(a=Q({inputs:{x:a},backend:r,attrs:{shape:e}}),y.push(a))}if(i!=null){let e=bM(i.shape,m);e!=null&&(i=Q({inputs:{x:i},backend:r,attrs:{shape:e}}),y.push(i))}let b=Q({inputs:{x:t},backend:r,attrs:{shape:[1,h,g(t.shape)/h]}});y.push(b);let x=new yM(v,n),S=[e.shape,[n.padInfo.top,n.padInfo.left],[n.strideHeight,n.strideWidth],[n.dilationHeight,n.dilationWidth],[n.inChannels],[n.filterWidth*n.inChannels],[n.outWidth]],C=r.runWebGLProgram(x,[e],`float32`,S),w=Q({inputs:{x:C},backend:r,attrs:{shape:v}});y.push(C),y.push(w);let T=i!=null,E=a!=null,D=s===`leakyrelu`,O=s?mA(s,!0):null,ee=new hA(m?w.shape:b.shape,m?b.shape:w.shape,m?[n.batchSize,_,n.outChannels]:[n.batchSize,n.outChannels,_],!0,!1,T,O,E,D),k=m?[w,b]:[b,w];if(i&&k.push(i),E&&k.push(a),D){let e=r.makeTensorInfo([],`float32`,Vr(o,`float32`));k.push(e),y.push(e)}let A=r.runWebGLProgram(ee,k,`float32`),te=Q({inputs:{x:A},backend:r,attrs:{shape:n.outShape}});y.push(A);for(let e of y)r.disposeIntermediateTensorInfo(e);return te}function CM(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dataFormat:c,dilations:l,dimRoundingMode:u}=r,d=Ec(c),f=pc(i.shape,a.shape,o,l,s,u,!1,d),p;if(f.filterHeight===1&&f.filterWidth===1&&f.dilationHeight===1&&f.dilationWidth===1&&f.strideHeight===1&&f.strideWidth===1&&(f.padInfo.type===`SAME`||f.padInfo.type===`VALID`))p=xM({x:i,filter:a,convInfo:f,backend:n});else if(f.strideWidth<=2&&d===`channelsLast`&&N().getBool(`WEBGL_EXP_CONV`)){let e=new vM(f),t=[[f.padInfo.top,f.padInfo.left],[f.strideHeight,f.strideWidth],[f.dilationHeight,f.dilationWidth],[f.inHeight,f.inWidth]];p=n.runWebGLProgram(e,[i,a],`float32`,t)}else if(N().getBool(`WEBGL_CONV_IM2COL`))p=SM({x:i,filter:a,convInfo:f,backend:n});else{let e=new gM(f);p=n.runWebGLProgram(e,[i,a],`float32`)}let m=Q({inputs:{x:p},backend:n,attrs:{shape:f.outShape}});return n.disposeIntermediateTensorInfo(p),m}var wM={kernelName:Ze,backendName:`webgl`,kernelFunc:CM},TM=class{constructor(e){this.variableNames=[`x`,`dy`],this.outputShape=e.filterShape;let t=e.strideHeight,n=e.strideWidth,r=e.padInfo.top,i=e.padInfo.left,a=e.dataFormat===`channelsLast`;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int d2 = coords.w;

        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yR = 0; yR < ${e.outHeight}; yR++) {
            int xR = wR + yR * ${t} - ${r};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${e.outWidth}; yC++) {
              int xC = wC + yC * ${n} - ${i};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              if (${a}) {
                float dyValue = getDy(b, yR, yC, d2);
                float xValue = getX(b, xR, xC, d1);
                dotProd += (xValue * dyValue);
              } else {
                float dyValue = getDy(b, d2, yR, yC);
                float xValue = getX(b, d1, xR, xC);
                dotProd += (xValue * dyValue);
              }

            }
          }
        }
        setOutput(dotProd);
      }
    `}},EM=class{constructor(e){this.variableNames=[`dy`,`W`],this.outputShape=e.inShape;let t=e.filterHeight,n=e.filterWidth,r=e.strideHeight,i=e.strideWidth,a=e.dataFormat===`channelsLast`,o=t-1-e.padInfo.top,s=n-1-e.padInfo.left,c=a?1:2,l=a?2:3,u=a?3:1;this.userCode=`
      const ivec2 pads = ivec2(${o}, ${s});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[${u}];

        ivec2 dyCorner = ivec2(coords[${c}], coords[${l}]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / ${r}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${n}; wC++) {
            float dyC = float(dyCCorner + wC) / ${i}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${n} - 1 - wC;

            for (int d2 = 0; d2 < ${e.outChannels}; d2++) {

              if (${a}) {
                float xValue = getDy(batch, idyR, idyC, d2);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              } else {
                float xValue = getDy(batch, d2, idyR, idyC);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }

            }
          }
        }
        setOutput(dotProd);
      }
    `}},DM=class{constructor(e){this.variableNames=[`x`,`dy`],this.outputShape=e.filterShape;let t=e.strideDepth,n=e.strideHeight,r=e.strideWidth,i=e.padInfo.front,a=e.padInfo.top,o=e.padInfo.left;this.userCode=`
      void main() {
        ivec5 coords = getOutputCoords();
        int wF = coords.x;
        int wR = coords.y;
        int wC = coords.z;
        int d1 = coords.w;
        int d2 = coords.u;

        float dotProd = 0.0;

        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yF = 0; yF < ${e.outDepth}; yF++) {
            int xF = wF + yF * ${t} - ${i};

            if (xF < 0 || xF >= ${e.inDepth}) {
              continue;
            }

            for (int yR = 0; yR < ${e.outHeight}; yR++) {
              int xR = wR + yR * ${n} - ${a};

              if (xR < 0 || xR >= ${e.inHeight}) {
                continue;
              }

              for (int yC = 0; yC < ${e.outWidth}; yC++) {
                int xC = wC + yC * ${r} - ${o};

                if (xC < 0 || xC >= ${e.inWidth}) {
                  continue;
                }

                float dyValue = getDy(b, yF, yR, yC, d2);
                float xValue = getX(b, xF, xR, xC, d1);
                dotProd += (xValue * dyValue);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}},OM=class{constructor(e){this.variableNames=[`dy`,`W`],this.outputShape=e.inShape;let t=e.filterDepth,n=e.filterHeight,r=e.filterWidth,i=e.strideDepth,a=e.strideHeight,o=e.strideWidth,s=t-1-e.padInfo.front,c=n-1-e.padInfo.top,l=r-1-e.padInfo.left;this.userCode=`
      const ivec3 pads = ivec3(${s}, ${c}, ${l});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.u;


        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyFCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        float dotProd = 0.0;
        for (int wF = 0; wF < ${t}; wF++) {
          float dyF = float(dyFCorner + wF) / ${i}.0;

          if (dyF < 0.0 || dyF >= ${e.outDepth}.0 || fract(dyF) > 0.0) {
            continue;
          }
          int idyF = int(dyF);

          int wFPerm = ${t} - 1 - wF;

          for (int wR = 0; wR < ${n}; wR++) {
            float dyR = float(dyRCorner + wR) / ${a}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
              fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            int wRPerm = ${n} - 1 - wR;

            for (int wC = 0; wC < ${r}; wC++) {
              float dyC = float(dyCCorner + wC) / ${o}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              int wCPerm = ${r} - 1 - wC;

              for (int d2 = 0; d2 < ${e.outChannels}; d2++) {
                float xValue = getDy(batch, idyF, idyR, idyC, d2);
                float wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}};function kM(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,dataFormat:c,dimRoundingMode:l,filterShape:u}=r,d=Ec(c),f=new TM(pc(i.shape,u,o,1,s,l,!1,d));return n.runWebGLProgram(f,[i,a],`float32`)}var AM={kernelName:Qe,backendName:`webgl`,kernelFunc:kM};function jM(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{inputShape:o,strides:s,pad:c,dataFormat:l,dimRoundingMode:u}=r,d=Ec(l),f=new EM(pc(o,a.shape,s,1,c,u,!1,d));return n.runWebGLProgram(f,[i,a],`float32`)}var MM={kernelName:$e,backendName:`webgl`,kernelFunc:jM};function NM(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r,l=new _M(mc(i.shape,a.shape,o,c,s));return n.runWebGLProgram(l,[i,a],`float32`)}var PM={kernelName:et,backendName:`webgl`,kernelFunc:NM};function FM(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,filterShape:c}=r,l=new DM(mc(i.shape,c,o,1,s));return n.runWebGLProgram(l,[i,a],`float32`)}var IM={kernelName:tt,backendName:`webgl`,kernelFunc:FM};function LM(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{pad:o,strides:s,inputShape:c}=r,l=new OM(mc(c,a.shape,s,1,o));return n.runWebGLProgram(l,[i,a],`float32`)}var RM={kernelName:nt,backendName:`webgl`,kernelFunc:LM},zM={kernelName:`Cos`,backendName:`webgl`,kernelFunc:Z({opSnippet:fA+`
  return cos(x);
`})},BM={kernelName:rt,backendName:`webgl`,kernelFunc:Z({opSnippet:`
  float e2x = exp(-x);
  return (e2x + 1.0 / e2x) / 2.0;
`})},VM=class{constructor(e,t,n,r,i){this.variableNames=[`Image`,`Boxes`,`BoxInd`],this.outputShape=[];let[a,o,s,c]=e,[l]=t,[u,d]=n;this.outputShape=[l,u,d,c];let f=+(r===`bilinear`),[p,m]=[`${o-1}.0`,`${s-1}.0`],[h,g,_]=u>1?[`${(o-1)/(u-1)}`,`(y2-y1) * height_ratio`,`y1*${p} + float(y)*(height_scale)`]:[`0.0`,`0.0`,`0.5 * (y1+y2) * ${p}`],[v,y,b]=d>1?[`${(s-1)/(d-1)}`,`(x2-x1) * width_ratio`,`x1*${m} + float(x)*(width_scale)`]:[`0.0`,`0.0`,`0.5 * (x1+x2) * ${m}`];this.userCode=`
      const float height_ratio = float(${h});
      const float width_ratio = float(${v});
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int y = coords[1];
        int x = coords[2];
        int d = coords[3];

        // get box vals
        float y1 = getBoxes(b,0);
        float x1 = getBoxes(b,1);
        float y2 = getBoxes(b,2);
        float x2 = getBoxes(b,3);

        // get image in batch index
        int bInd = round(getBoxInd(b));
        if(bInd < 0 || bInd >= ${a}) {
          return;
        }

        float height_scale = ${g};
        float width_scale = ${y};

        float in_y = ${_};
        if( in_y < 0.0 || in_y > ${p} ) {
          setOutput(float(${i}));
          return;
        }
        float in_x = ${b};
        if( in_x < 0.0 || in_x > ${m} ) {
          setOutput(float(${i}));
          return;
        }

        vec2 sourceFracIndexCR = vec2(in_x,in_y);
        if(${f} == 1) {
          // Compute the four integer indices.
          ivec2 sourceFloorCR = ivec2(sourceFracIndexCR);
          ivec2 sourceCeilCR = ivec2(ceil(sourceFracIndexCR));

          float topLeft = getImage(b, sourceFloorCR.y, sourceFloorCR.x, d);
          float bottomLeft = getImage(b, sourceCeilCR.y, sourceFloorCR.x, d);
          float topRight = getImage(b, sourceFloorCR.y, sourceCeilCR.x, d);
          float bottomRight = getImage(b, sourceCeilCR.y, sourceCeilCR.x, d);

          vec2 fracCR = sourceFracIndexCR - vec2(sourceFloorCR);

          float top = topLeft + (topRight - topLeft) * fracCR.x;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          float newValue = top + (bottom - top) * fracCR.y;
          setOutput(newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          ivec2 sourceNearestCR = ivec2(floor(
            sourceFracIndexCR + vec2(0.5,0.5)));
          float newValue = getImage(b, sourceNearestCR.y, sourceNearestCR.x, d);
          setOutput(newValue);
        }
      }
    `}},HM={kernelName:ot,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{image:i,boxes:a,boxInd:o}=t,{cropSize:s,method:c,extrapolationValue:l}=r,u=new VM(i.shape,a.shape,s,c,l);return n.runWebGLProgram(u,[i,a,o],`float32`)}},UM;(function(e){e.Prod=`*`,e.Sum=`+`})(UM||={});var WM=class{constructor(e,t,n,r){this.op=e,this.outputShape=t,this.variableNames=[`x`],this.customUniforms=[{name:`index`,type:`float`}];let i=this.outputShape.length,a=this.op===UM.Prod?`1.0`:`0.0`,o=n?a:`getX(${GM(i,`coords`,this.op)})`,s=this.outputShape[this.outputShape.length-1],c=``,l=``;n?(c=r?`end != ${s-1}`:`end != 0`,l=r?`end + 1`:`end - 1`):(c=r?`end + pow2 < ${s}`:`end >= pow2`,l=r?`end + pow2`:`end - pow2`),this.userCode=`
      void main() {
        ${X(i)} coords = getOutputCoords();
        int end = ${KM(i,`coords`,this.op)};
        float val = ${o};
        int pow2 = int(pow(2.0, index));
        if (${c}) {
          int idx = ${l};
          ${KM(i,`coords`,this.op)} = idx;
          val ${this.op}= getX(${GM(i,`coords`,this.op)});
        }
        setOutput(val);
      }
    `}};function GM(e,t,n){if(e===1)return`${t}`;if(e===2)return`${t}.x, ${t}.y`;if(e===3)return`${t}.x, ${t}.y, ${t}.z`;if(e===4)return`${t}.x, ${t}.y, ${t}.z, ${t}.w`;throw Error(`Cumulative ${n} for rank ${e} is not yet supported`)}function KM(e,t,n){if(e===1)return`${t}`;if(e===2)return`${t}.y`;if(e===3)return`${t}.z`;if(e===4)return`${t}.w`;throw Error(`Cumulative ${n} for rank ${e} is not yet supported`)}function qM(e,t,n,r,i,a){let o=t.shape.length,s=_u([r],o),c=t;s!=null&&(c=PA({inputs:{x:t},backend:n,attrs:{perm:s}}));let l=yu(1,o)[0];if(l!==o-1)throw Error(`WebGL cumprod shader expects an inner-most axis=${t.shape.length-1} but got axis=${r}`);let u=c.shape[l],d=eA({inputs:{x:c},backend:n});for(let t=0;t<=Math.ceil(Math.log2(u))-1;t++){let r=new WM(e,c.shape,!1,a),i=[[t]],o=d;d=n.runWebGLProgram(r,[d],d.dtype,i),n.disposeIntermediateTensorInfo(o)}if(i){let t=new WM(e,c.shape,i,a),r=d;d=n.runWebGLProgram(t,[d],d.dtype),n.disposeIntermediateTensorInfo(r)}if(s!=null){let e=vu(s),t=PA({inputs:{x:d},backend:n,attrs:{perm:e}});return n.disposeIntermediateTensorInfo(d),n.disposeIntermediateTensorInfo(c),t}return d}function JM(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r;return qM(UM.Prod,i,n,a,o,s)}var YM={kernelName:it,backendName:`webgl`,kernelFunc:JM};function XM(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r;return qM(UM.Sum,i,n,a,o,s)}var ZM={kernelName:at,backendName:`webgl`,kernelFunc:XM};function QM(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,weights:a}=t,{size:o,binaryOutput:s}=r;if(i.shape.length===1){let e=bO(n.readSync(i.dataId),n.readSync(a.dataId),a.dtype,a.shape,o);return n.makeTensorInfo([o],a.dtype,e)}if(i.shape.length===2){let e=xO(n.bufferSync(i),n.bufferSync(a),o,s);return n.makeTensorInfo(e.shape,a.dtype,e.values)}throw Error(`Error in denseBincount: input must be at most rank 2, but got rank${i.shape.length}.`)}var $M={kernelName:st,backendName:`webgl`,kernelFunc:QM},eN=class{constructor(e,t,n){this.variableNames=[`x`],this.outputShape=[],this.outputShape=e,this.blockSize=t,this.dataFormat=n,this.userCode=`
    void main() {
      ivec4 coords = getOutputCoords();
      int b = coords[0];
      int h = ${this.getHeightCoordString()};
      int w = ${this.getWidthCoordString()};
      int d = ${this.getDepthCoordString()};

      int in_h = h / ${t};
      int offset_h = imod(h, ${t});
      int in_w = w / ${t};
      int offset_w = imod(w, ${t});
      int offset_d = (offset_h * ${t} + offset_w) *
        ${this.getOutputDepthSize()};
      int in_d = d + offset_d;

      float result = ${this.getInputSamplingString()};
      setOutput(result);
    }
  `}getHeightCoordString(){return this.dataFormat===`NHWC`?`coords[1]`:`coords[2]`}getWidthCoordString(){return this.dataFormat===`NHWC`?`coords[2]`:`coords[3]`}getDepthCoordString(){return this.dataFormat===`NHWC`?`coords[3]`:`coords[1]`}getOutputDepthSize(){return this.dataFormat===`NHWC`?this.outputShape[3]:this.outputShape[1]}getInputSamplingString(){return this.dataFormat===`NHWC`?`getX(b, in_h, in_w, in_d)`:`getX(b, in_d, in_h, in_w)`}};function tN(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockSize:a,dataFormat:o}=r,s=i.shape[0],c=o===`NHWC`?i.shape[1]:i.shape[2],l=o===`NHWC`?i.shape[2]:i.shape[3],u=o===`NHWC`?i.shape[3]:i.shape[1],d=c*a,f=l*a,p=u/(a*a),m=new eN(o===`NHWC`?[s,d,f,p]:[s,p,d,f],a,o);return n.runWebGLProgram(m,[i],i.dtype)}var nN={kernelName:ct,backendName:`webgl`,kernelFunc:tN},rN=class{constructor(e,t=!1,n=null,r=!1,i=!1){this.variableNames=[`x`,`W`],this.customUniforms=[{name:`pads`,type:`ivec2`},{name:`strides`,type:`ivec2`},{name:`dilations`,type:`ivec2`},{name:`inDims`,type:`ivec2`}],this.outputShape=e.outShape,this.enableShapeUniforms=HD(this.outputShape.length);let a=e.filterHeight,o=e.filterWidth,s=e.outChannels/e.inChannels,c=``,l=``;n&&(c=r?`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${n}
        }`:i?`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${n}
        }`:`
          float activation(float x) {
            ${n}
          }
        `,l=`result = activation(result);`);let u=t?`result += getBiasAtOutCoords();`:``;t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),i&&this.variableNames.push(`leakyreluAlpha`),this.userCode=`
      ${c}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${s};
        int q = d2 - d1 * ${s};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        // TO DO(dsmilkov): Flatten the two for loops and vec4 the operations.
        for (int wR = 0; wR < ${a}; wR++) {
          int xR = xRCorner + wR * dilations[0];

          if (xR < 0 || xR >= inDims[0]) {
            continue;
          }

          for (int wC = 0; wC < ${o}; wC++) {
            int xC = xCCorner + wC * dilations[1];

            if (xC < 0 || xC >= inDims[1]) {
              continue;
            }

            float xVal = getX(batch, xR, xC, d1);
            float wVal = getW(wR, wC, d1, q);
            dotProd += xVal * wVal;
          }
        }

        float result = dotProd;
        ${u}
        ${l}
        setOutput(result);
      }
    `}},iN=class{constructor(e,t=!1,n=null,r=!1,i=!1){this.variableNames=[`x`,`W`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`pads`,type:`ivec2`},{name:`strides`,type:`ivec2`},{name:`dilations`,type:`ivec2`},{name:`inDims`,type:`ivec2`}],this.outputShape=e.outShape,this.enableShapeUniforms=HD(this.outputShape.length);let a=e.outChannels/e.inChannels,o=e.padInfo.left,s=e.strideWidth,c=e.dilationWidth,u=e.filterHeight,d=e.filterWidth,f=d,p=`
      int xR; int xC; int xCOffset;
      vec4 wTexel; vec4 previous; vec4 final;`;for(let e=0;e<d;e++)p+=`
          vec4 xTexelC${e*2};
          int xTexelC${e*2}Ready;
          vec4 xTexelC${e*2+1};
          int xTexelC${e*2+1}Ready;
          vec4 xC${e};`;p+=`
    for (int r = 0; r < ${u}; r++) {
      `;for(let e=0;e<d;e++)p+=`
          xTexelC${e*2} = vec4(0.0);
          xTexelC${e*2}Ready = 0;
          xTexelC${e*2+1} = vec4(0.0);
          xTexelC${e*2+1}Ready = 0;
          xC${e} = vec4(0.0);`;p+=`
        xR = xRCorner + r * dilations[0];
        if (xR >=0 && xR < inDims[0]) {
      `;for(let e=0;e<(f+1)/2;e++){let t=e*2;if(p+=`
          xC = xCCorner + ${t*c};
          `,s===1){if(t<d&&(o%2==1?(p+=`
                xCOffset = xC + 1;
                if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }
              `,p+=c===1&&t>0?`
                xC${t} = vec4(xTexelC${t-2}.zw, xTexelC${t}.xy);
                `:`
                  xCOffset = xC + 1 - 2;

                  if (xCOffset >= 0 && xCOffset < inDims[1]) {
                    previous = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      previous.zw = vec2(0.0);
                    }

                    xC${t} = vec4(previous.zw, xTexelC${t}.xy);
                  } else {
                    xC${t} = vec4(0.0, 0.0, xTexelC${t}.xy);
                  }
                  `):p+=`
                if (xC >= 0 && xC < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }

                xC${t} = xTexelC${t};
                `,t+1<d)){let e=o%2==0?l(c):c;c%2==0&&o%2==1||c%2!=0&&o%2!=1?(p+=`
                  xCOffset = xC + imod(pads[1], 2) + ${e};

                  if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t+1}Ready == 0) {
                    xTexelC${t+1} = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      xTexelC${t+1}.zw = vec2(0.0);
                    }
                    xTexelC${t+1}Ready = 1;
                  }
                  `,p+=c>1?`
                    xCOffset -= 2;
                    if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);
                     xC${t+1} = vec4(previous.zw, xTexelC${t+1}.xy);
                    } else {
                     xC${t+1} = vec4(0.0, 0.0, xTexelC${t+1}.xy);
                    }
                    `:`
                    xC${t+1} = vec4(xTexelC${t}.zw, xTexelC${t+1}.xy);
                    `):p+=e===1?`
                    xC${t+1} = xTexelC${t};
                    `:`
                    xCOffset = xC + ${e};

                    if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t+1}Ready == 0) {
                      xTexelC${t+1} = getX(batch, xR, xCOffset, d1);
                      if (xCOffset + 1 >= inDims[1]) {
                        xTexelC${t+1}.zw = vec2(0.0);
                      }
                      xTexelC${t+1}Ready = 1;
                    }

                    xC${t+1} = xTexelC${t+1};
                    `}}else t<d&&(o%2==1?(p+=`
                xCOffset = xC + 1 - strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xCOffset, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }

                if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${t+1}Ready == 0) {
                  xTexelC${t+1} = getX(batch, xR, xC + 1, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xC + 2 >= inDims[1]) {
                    xTexelC${t+1}.zw = vec2(0.0);
                  }
                  xTexelC${t+1}Ready = 1;
                }

                xC${t} = vec4(xTexelC${t}.zw, xTexelC${t+1}.zw);
              `,t+1<d&&(p+=`
                  final = vec4(0.0);
                  xCOffset = xC + 1 + strides[1];
                  if(xCOffset >= 0 && xCOffset < inDims[1]) {
                    final = getX(batch, xR, xCOffset, d1);
                  }
                  xC${t+1} = vec4(xTexelC${t+1}.xy, final.xy);
                `)):(p+=`
                if(xC >= 0 && xC < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }

                xCOffset = xC + strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t+1}Ready == 0) {
                  xTexelC${t+1} = getX(batch, xR, xCOffset, d1);
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${t+1}.zw = vec2(0.);
                  }
                  xTexelC${t+1}Ready = 1;
                }

                xC${t} = vec4(
                  xTexelC${t}.xy, xTexelC${t+1}.xy);
              `,t+1<d&&(p+=`
                  xC${t+1} = vec4(xTexelC${t}.zw, xTexelC${t+1}.zw);
                `)));t<d&&(p+=`
            wTexel = getW(r, ${t}, d1, q);
            dotProd += xC${t} * vec4(wTexel.xz, wTexel.xz);
          `,t+1<d&&(p+=`
              wTexel = getW(r, ${t+1}, d1, q);
              dotProd += xC${t+1} * vec4(wTexel.xz, wTexel.xz);
            `))}p+=`
    }
  `,p+=`
      }
    `;let m=``,h=``;n&&(m=r?`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${n}
        }`:i?`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${n}
        }`:`vec4 activation(vec4 x) {
          ${n}
        }`,h=`result = activation(result);`);let g=t?`result += getBiasAtOutCoords();`:``;t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),i&&this.variableNames.push(`leakyreluAlpha`),this.userCode=`
      ${m}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${a};
        int q = d2 - d1 * ${a};
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
        vec4 dotProd = vec4(0.000000000000001);

        ${p}

        vec4 result = dotProd - vec4(0.000000000000001);
        ${g}
        ${h}
        setOutput(result);
      }
    `}};function aN(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c,dimRoundingMode:l}=r,u=c;u??=[1,1],f(Tc(o,u),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${o} and dilations '${u}'`);let d=pc(i.shape,a.shape,o,u,s,l,!0),p;p=N().getBool(`WEBGL_PACK_DEPTHWISECONV`)&&d.strideWidth<=2&&d.outChannels/d.inChannels===1?new iN(d):new rN(d);let m=[[d.padInfo.top,d.padInfo.left],[d.strideHeight,d.strideWidth],[d.dilationHeight,d.dilationWidth],[d.inHeight,d.inWidth]];return n.runWebGLProgram(p,[i,a],`float32`,m)}var oN={kernelName:lt,backendName:`webgl`,kernelFunc:aN},sN=class{constructor(e){this.variableNames=[`x`,`dy`],this.outputShape=e.filterShape;let t=e.strideHeight,n=e.strideWidth,r=e.padInfo.top,i=e.padInfo.left,a=e.outChannels/e.inChannels;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int dm = coords.w;
        int d2 = d1 * ${a} + dm;

        float dotProd = 0.0;

        // TO DO: Vec4 over the batch size
        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yR = 0; yR < ${e.outHeight}; yR++) {
            int xR = wR + yR * ${t} - ${r};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${e.outWidth}; yC++) {
              int xC = wC + yC * ${n} - ${i};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);
            }
          }
        }
        setOutput(dotProd);
      }
    `}},cN=class{constructor(e){this.variableNames=[`dy`,`W`],this.outputShape=e.inShape;let t=e.filterHeight,n=e.filterWidth,r=e.strideHeight,i=e.strideWidth,a=t-1-e.padInfo.top,o=n-1-e.padInfo.left,s=e.outChannels/e.inChannels;this.userCode=`
      const ivec2 pads = ivec2(${a}, ${o});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];
        ivec2 dyCorner = coords.yz - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        float dotProd = 0.0;

        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / ${r}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${n}; wC++) {
            float dyC = float(dyCCorner + wC) / ${i}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${n} - 1 - wC;

            // TO DO: Vec4 over the channelMul
            for (int dm = 0; dm < ${s}; dm++) {
              int d2 = d1 * ${s} + dm;
              float xValue = getDy(batch, idyR, idyC, d2);
              float wValue = getW(wRPerm, wCPerm, d1, dm);
              dotProd += xValue * wValue;
            }
          }
        }
        setOutput(dotProd);
      }
    `}};function lN(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,dilations:s,pad:c,dimRoundingMode:l,filterShape:u}=r,d=new sN(pc(i.shape,u,o,s,c,l,!0));return n.runWebGLProgram(d,[i,a],`float32`)}var uN={kernelName:ut,backendName:`webgl`,kernelFunc:lN};function dN(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{strides:o,dilations:s,pad:c,dimRoundingMode:l,inputShape:u}=r,d=new cN(pc(u,a.shape,o,s,c,l,!0));return n.runWebGLProgram(d,[i,a],`float32`)}var fN={kernelName:dt,backendName:`webgl`,kernelFunc:dN},pN=class{constructor(e){this.variableNames=[`X`],this.outputShape=[e,e],this.userCode=`
      void main() {
          ivec2 coords = getOutputCoords();
          float val = coords[0] == coords[1] ? getX(coords[0]) : 0.0;
          setOutput(val);
      }
    `}};function mN(e){let{inputs:t,backend:n}=e,{x:r}=t,i=[...r.shape,...r.shape],a=g(r.shape),o=Q({inputs:{x:r},backend:n,attrs:{shape:[a]}}),s=new pN(a),c=n.runWebGLProgram(s,[o],o.dtype),l=Q({inputs:{x:c},backend:n,attrs:{shape:i}});return n.disposeIntermediateTensorInfo(o),n.disposeIntermediateTensorInfo(c),l}var hN={kernelName:ft,backendName:`webgl`,kernelFunc:mN},gN=class{constructor(e){this.variableNames=[`x`,`W`],this.outputShape=e.outShape;let{inHeight:t,inWidth:n,padInfo:r,strideHeight:i,strideWidth:a,filterHeight:o,filterWidth:s,dilationHeight:c,dilationWidth:l}=e,{top:u,left:d}=r;this.userCode=`
      const ivec2 strides = ivec2(${i}, ${a});
      const ivec2 pads = ivec2(${u}, ${d});
      const float neg_infinity = -3.4e38;

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.w;
        ivec2 outTopLeftCorner =
            coords.yz * strides - pads;
        int hBeg = outTopLeftCorner.x;
        int wBeg = outTopLeftCorner.y;

        float curVal = neg_infinity;
        for (int h = 0; h < ${o}; h++) {
          int hIn = hBeg + h * ${c};

          if (hIn >= 0 && hIn < ${t}) {
            for (int w = 0; w < ${s}; w++) {
              int wIn = wBeg + w * ${l};

              if (wIn >= 0 && wIn < ${n}) {
                float xVal = getX(batch, hIn, wIn, d1);
                float wVal = getW(h, w, d1);

                float val = xVal + wVal;
                if (val > curVal) {
                  curVal = val;
                }
              }
            }
          }
        }

        float result = curVal;
        setOutput(result);
      }
    `}};function _N(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r,l=uc(i.shape,a.shape,o,s,`NHWC`,c),u,d=new gN(l);u=n.runWebGLProgram(d,[i,a],`float32`);let f=Q({inputs:{x:u},backend:n,attrs:{shape:l.outShape}});return n.disposeIntermediateTensorInfo(u),f}var vN={kernelName:pt,backendName:`webgl`,kernelFunc:_N};function yN(e){let{inputs:t,backend:n,attrs:r}=e,{equation:i}=r,a=t,{allDims:o,summedDims:s,idDims:c}=Y_(i,a.length);Z_(o.length,c,a);let{path:l,steps:u}=Q_(s,c),d=u.length,f=null,p=o.length,m=[];for(let e=0;e<d;++e){for(let t of u[e]){let{permutationIndices:e,expandDims:r}=X_(p,c[t]),i;$_(e)?i=a[t]:(i=PA({inputs:{x:a[t]},backend:n,attrs:{perm:e}}),m.push(i));let o=i.shape.slice();for(let e=0;e<r.length;++e)o.splice(r[e],0,1);_(i.shape,o)||(i=Q({inputs:{x:i},backend:n,attrs:{shape:o}}),m.push(i)),f===null?f=i:(f=yA({inputs:{a:i,b:f},backend:n}),m.push(f))}e<d-1&&(l[e]>=0&&(f=MA({inputs:{x:f},backend:n,attrs:{axis:l[e]-(o.length-p),keepDims:!1}}),m.push(f)),p--)}for(let e of m)e!==f&&n.disposeIntermediateTensorInfo(e);return f}var bN={kernelName:_t,backendName:`webgl`,kernelFunc:yN},xN={kernelName:`Elu`,backendName:`webgl`,kernelFunc:Z({opSnippet:`return (x >= 0.0) ? x : (exp(x) - 1.0);`,packedOpSnippet:`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`})},SN=`return (b >= 1.0) ? a : a * (b + 1.0);`,CN=`
  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));
  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));
`,wN={kernelName:vt,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n}=e,{dy:r,y:i}=t,a=N().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)?new $k(CN,r.shape,i.shape):new Zk(SN,r.shape,i.shape);return n.runWebGLProgram(a,[r,i],r.dtype)}},TN={kernelName:yt,backendName:`webgl`,kernelFunc:pA({opSnippet:`return float(a == b);`,packedOpSnippet:`
  return vec4(equal(a, b));
`,dtype:`bool`,cpuKernelImpl:TO})},EN={kernelName:`Erf`,backendName:`webgl`,kernelFunc:Z({opSnippet:`
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  float p = ${j_};
  float a1 = ${M_};
  float a2 = ${N_};
  float a3 = ${P_};
  float a4 = ${F_};
  float a5 = ${I_};

  float sign = sign(x);
  x = abs(x);
  float t = 1.0 / (1.0 + p * x);
  return sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x));
`})},DN=Z({opSnippet:fA+`
  return exp(x);
`,packedOpSnippet:`
  vec4 result = exp(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,cpuKernelImpl:EO,dtype:`float32`}),ON={kernelName:`Exp`,backendName:`webgl`,kernelFunc:DN};function kN(e){let{inputs:t,attrs:n,backend:r}=e,{dim:i}=n,{input:a}=t,o=a.shape.length,s=a.shape.slice(),c=i;return i<0&&(f(-(o+1)<=i,()=>`Axis must be in the interval [${-(o+1)}, ${o}]`),c=o+i+1),s.splice(c,0,1),Q({inputs:{x:a},backend:r,attrs:{shape:s}})}var AN={kernelName:bt,backendName:`webgl`,kernelFunc:kN},jN=`return exp(x) - 1.0;`,MN={kernelName:xt,backendName:`webgl`,kernelFunc:Z({opSnippet:jN,packedOpSnippet:jN,cpuKernelImpl:DO})},NN=class{constructor(e,t,n){this.variableNames=[`real`,`imag`];let r=t[1];this.outputShape=t;let i=n?`2.0 * ${Math.PI}`:`-2.0 * ${Math.PI}`,a=n?`${r}.0`:`1.0`,o;if(e===`real`)o=`return real * expR - imag * expI;`;else if(e===`imag`)o=`return real * expI + imag * expR;`;else throw Error(`FFT component must be either "real" or "imag", got ${e}.`);this.userCode=`
      const float exponentMultiplier = ${i};

      float unaryOpComplex(float real, float expR, float imag, float expI) {
        ${o}
      }

      float mulMatDFT(int batch, int index) {
        float indexRatio = float(index) / float(${r});
        float exponentMultiplierTimesIndexRatio =
            exponentMultiplier * indexRatio;

        float result = 0.0;

        for (int i = 0; i < ${r}; i++) {
          // x = (-2|2 * PI / N) * index * i;
          float x = exponentMultiplierTimesIndexRatio * float(i);
          float expR = cos(x);
          float expI = sin(x);
          float real = getReal(batch, i);
          float imag = getImag(batch, i);

          result +=
              unaryOpComplex(real, expR, imag, expI) / ${a};
        }

        return result;
      }

      void main() {
        ivec2 coords = getOutputCoords();
        setOutput(mulMatDFT(coords[0], coords[1]));
      }
    `}};function PN(e,t,n){let r=n.texData.get(e.dataId),i=g(e.shape),a=e.shape[e.shape.length-1],o=i/a,s=Q({inputs:{x:e},backend:n,attrs:{shape:[o,a]}}),c=s.shape,l=new NN(`real`,c,t),u=new NN(`imag`,c,t),d=[{dataId:r.complexTensorInfos.real.dataId,dtype:r.complexTensorInfos.real.dtype,shape:c},{dataId:r.complexTensorInfos.imag.dataId,dtype:r.complexTensorInfos.imag.dtype,shape:c}],f=n.runWebGLProgram(l,d,`float32`),p=n.runWebGLProgram(u,d,`float32`),m=nA({inputs:{real:f,imag:p},backend:n});n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(p);let h=Q({inputs:{x:m},backend:n,attrs:{shape:e.shape}});return n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(m),h}function FN(e){let{inputs:t,backend:n}=e,{input:r}=t;return PN(r,!1,n)}var IN={kernelName:`FFT`,backendName:`webgl`,kernelFunc:FN},LN=class{constructor(e,t){this.outputShape=[],this.customUniforms=[{name:`value`,type:`float`}],this.variableNames=[`x`],this.outputShape=e,this.userCode=`
      void main() {
        // Input can be obtained from uniform value.
        setOutput(value);
      }
    `}};function RN(e){let{backend:t,attrs:n}=e,{shape:r,value:i}=n,{dtype:a}=n;if(a||=ae(i),a===`string`){let e=E(a,g(r));return e.fill(i),t.makeTensorInfo(r,a,e)}{let e=new LN(r,i),n=[[i]];return t.runWebGLProgram(e,[],a,n)}}var zN={kernelName:St,backendName:`webgl`,kernelFunc:RN},BN=class{constructor(e){this.variableNames=[`Image`],this.outputShape=[];let t=e[2];this.outputShape=e,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];

          int coordX = ${t} - x - 1;
          float outputValue;
          if(coordX >= 0 && coordX < ${t}) {
            outputValue = getImage(coords[0], coords[1], coordX, coords[3]);
          } else {
            outputValue = getImage(coords[0], coords[1], coords[2], coords[3]);
          }
          setOutput(outputValue);
        }
    `}},VN={kernelName:Ct,backendName:`webgl`,kernelFunc:({inputs:e,backend:t})=>{let{image:n}=e,r=t,i=new BN(n.shape);return r.runWebGLProgram(i,[n],n.dtype)}},HN=`return floor(x);`,UN={kernelName:wt,backendName:`webgl`,kernelFunc:Z({opSnippet:HN,packedOpSnippet:HN,cpuKernelImpl:OO})},WN={kernelName:Tt,backendName:`webgl`,kernelFunc:pA({opSnippet:`
  float s = sign(a) * sign(b);
  int ia = round(a);
  int ib = round(b);
  if (ib != 0) {
    // Windows (D3D) wants guaranteed non-zero int division at compile-time.
    return float(idiv(ia, ib, s));
  } else {
    return NAN;
  }
`,packedOpSnippet:`
  ivec4 ia = round(a);
  ivec4 ib = round(b);
  bvec4 cond = notEqual(ib, ivec4(0));
  ivec4 result = ivec4(0);
  vec4 s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    result[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    result[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    result[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    result[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4(result);
`,dtype:`int32`})},GN=class{constructor(e){this.variableNames=[`A`];let t=RE(),[n,r]=e;this.outputShape=e,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${r}.0, ${n}.0);

        vec4 values = ${t.texture2D}(A, uv);
        float value;
        if (depth == 0) {
          value = values.r;
        } else if (depth == 1) {
          value = values.g;
        } else if (depth == 2) {
          value = values.b;
        } else if (depth == 3) {
          value = values.a;
        }

        setOutput(floor(value * 255.0 + 0.5));
      }
    `}},KN=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!1,this.packedOutput=!0;let t=RE(),[n,r]=e;this.outputShape=e,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];

        vec4 result = vec4(0.);

        for(int row=0; row<=1; row++) {
          for(int col=0; col<=1; col++) {
            texC = coords[1] + row;
            depth = coords[2] + col;

            vec2 uv = (vec2(texC, texR) + halfCR) /
                       vec2(${r}.0, ${n}.0);
            vec4 values = ${t.texture2D}(A, uv);
            float value;
            if (depth == 0) {
              value = values.r;
            } else if (depth == 1) {
              value = values.g;
            } else if (depth == 2) {
              value = values.b;
            } else if (depth == 3) {
              value = values.a;
            }

            result[row * 2 + col] = floor(value * 255.0 + 0.5);
          }
        }

        ${t.output} = result;
      }
    `}},qN={kernelName:lr,backendName:`webgl`,kernelFunc:XN},JN,YN=N().getBool(`CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU`);function XN(e){let{inputs:t,backend:n,attrs:r}=e,{pixels:i}=t,{numChannels:a}=r,o=typeof HTMLVideoElement<`u`&&i instanceof HTMLVideoElement,s=typeof HTMLImageElement<`u`&&i instanceof HTMLImageElement,[c,l]=o?[i.videoWidth,i.videoHeight]:[i.width,i.height],u=[l,c],d=[l,c,a];if(s||o){let e=N().getBool(`CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU`);(JN==null||e!==YN)&&(YN=e,JN=document.createElement(`canvas`).getContext(`2d`,{willReadFrequently:YN})),JN.canvas.width=c,JN.canvas.height=l,JN.drawImage(i,0,0,c,l),i=JN.canvas}let f=n.makeTensorInfo(u,`int32`);n.texData.get(f.dataId).usage=IT.PIXELS,n.gpgpu.uploadPixelDataToTexture(n.getTexture(f.dataId),i);let p=N().getBool(`WEBGL_PACK`)?new KN(d):new GN(d),m=n.runWebGLProgram(p,[f],`int32`);return n.disposeData(f.dataId),m}function ZN(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f,activation:p,leakyreluAlpha:m}=r,h=Ec(u),g=pc(i.shape,a.shape,c,d,l,f,!1,h),_,v=[],y=o!=null,b=s!=null,x=p===`leakyrelu`,S=()=>{let e=[i,a],t=(e,t)=>{if(t===`NCHW`&&e.shape.length===1&&e.shape[0]!==1){let t=Q({inputs:{x:e},backend:n,attrs:{shape:[e.shape[0],1,1]}});return v.push(t),t}return e};if(y&&e.push(t(o,u)),b&&e.push(t(s,u)),x){let t=n.makeTensorInfo([],`float32`,Vr(m,`float32`));e.push(t),v.push(t)}return e};if(g.filterHeight===1&&g.filterWidth===1&&g.dilationHeight===1&&g.dilationWidth===1&&g.strideHeight===1&&g.strideWidth===1&&(g.padInfo.type===`SAME`||g.padInfo.type===`VALID`))_=xM({x:i,filter:a,convInfo:g,backend:n,bias:o,activation:p,preluActivationWeights:s,leakyreluAlpha:m});else if(g.strideWidth<=2&&h===`channelsLast`&&N().getBool(`WEBGL_EXP_CONV`)){let e=new vM(g,y,p?mA(p,!0):null,b,x),t=[[g.padInfo.top,g.padInfo.left],[g.strideHeight,g.strideWidth],[g.dilationHeight,g.dilationWidth],[g.inHeight,g.inWidth]],r=S();_=n.runWebGLProgram(e,r,`float32`,t)}else if(N().getBool(`WEBGL_CONV_IM2COL`))_=SM({x:i,filter:a,convInfo:g,backend:n,bias:o,activation:p,preluActivationWeights:s,leakyreluAlpha:m});else{let e=new gM(g,y,p?mA(p,!1):null,b,x),t=S();_=n.runWebGLProgram(e,t,`float32`)}let C=Q({inputs:{x:_},backend:n,attrs:{shape:g.outShape}});return v.push(_),v.forEach(e=>n.disposeIntermediateTensorInfo(e)),C}var QN={kernelName:fr,backendName:`webgl`,kernelFunc:ZN};function $N(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dilations:u,dimRoundingMode:d,activation:p,leakyreluAlpha:m}=r,h=[],g=u;g??=[1,1],f(Tc(c,g),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${c} and dilations '${g}'`);let _=pc(i.shape,a.shape,c,g,l,d,!0),v=N().getBool(`WEBGL_PACK_DEPTHWISECONV`)&&_.strideWidth<=2&&_.outChannels/_.inChannels===1,y=p?mA(p,v):null,b=[i,a],x=o!=null,S=s!=null,C=p===`leakyrelu`;if(x&&b.push(o),S&&b.push(s),C){let e=n.makeTensorInfo([],`float32`,Vr(m,`float32`));b.push(e),h.push(e)}let w;w=v?new iN(_,x,y,S,C):new rN(_,x,y,S,C);let T=[[_.padInfo.top,_.padInfo.left],[_.strideHeight,_.strideWidth],[_.dilationHeight,_.dilationWidth],[_.inHeight,_.inWidth]],E=n.runWebGLProgram(w,b,`float32`,T);return h.forEach(e=>n.disposeIntermediateTensorInfo(e)),E}var eP={kernelName:pr,backendName:`webgl`,kernelFunc:$N},tP=class{constructor(e,t,n,r){this.sliceDim=e,this.strides=t,this.paramsShape=r,this.variableNames=[`x`,`indices`],this.outputShape=n;let i=X(n.length),a=`
    int index;`;for(let e=0;e<this.sliceDim;e++)a+=`
          index = round(getIndices(coords[0], ${e}));
          out_of_bounds = out_of_bounds || index < 0;
          out_of_bounds = out_of_bounds || index >= ${this.paramsShape[e]};
          flattenIndex += index * ${this.strides[e]};`;this.userCode=`
         void main() {
          ${i} coords = getOutputCoords();
          int flattenIndex = 0;
          bool out_of_bounds = false;

          ${a}

          setOutput(out_of_bounds ? 0.0 : getX(flattenIndex, coords[1]));
        }
      `}};function nP(e){let{inputs:t,backend:n}=e,{params:r,indices:i}=t,a=i.shape,o=a[a.length-1],s=g(r.shape),[c,l,u,d]=os(r,i),f=Q({inputs:{x:i},backend:n,attrs:{shape:[l,o]}}),p=Q({inputs:{x:r},backend:n,attrs:{shape:[g(r.shape)/u,u]}});if(n.shouldExecuteOnCPU([r,i])||r.dtype===`string`){let e=kO(n.readSync(i.dataId),n.bufferSync(r),r.dtype,l,o,u,d,r.shape,s);return n.makeTensorInfo(c,r.dtype,e.values)}let m=new tP(o,d,[l,u],r.shape),h=n.runWebGLProgram(m,[p,f],p.dtype),_=Q({inputs:{x:h},backend:n,attrs:{shape:c}});return n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(h),_}var rP={kernelName:Ot,backendName:`webgl`,kernelFunc:nP},iP=class{constructor(e,t){this.variableNames=[`A`,`indices`],this.outputShape=t,this.rank=t.length;let n=X(this.rank),r=aP(e,2);this.userCode=`
      void main() {
        ${n} resRC = getOutputCoords();
        int index = int(getIndices(resRC.x, resRC.z));
        float inBounds = (index >= 0) && (index < ${e[2]}) ? 1.0 : 0.0;
        setOutput(inBounds * getA(${r}));
      }
    `}};function aP(e,t){let n=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`],r=[];for(let t=0;t<e.length;t++)t===2?r.push(`index`):r.push(`${n[t]}`);return r.join()}function oP(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,indices:a}=t,{axis:o,batchDims:s}=r,c=C(o,i.shape)[0];if(N().get(`DEBUG`)){let e=n.readSync(a.dataId),t=i.shape[c];for(let n=0;n<e.length;++n){let r=e[n];f(r<=t-1&&r>=0,()=>`GatherV2: the index value ${r} is not in [0, ${t-1}]`)}}let l=_v(i,a,c,s),u=g(a.shape),d=[],p=Q({inputs:{x:i},backend:n,attrs:{shape:[l.batchSize,l.outerSize,l.dimSize,l.sliceSize]}}),m=Q({inputs:{x:a},backend:n,attrs:{shape:[l.batchSize,u/l.batchSize]}});d.push(p),d.push(m);let h=[l.batchSize,l.outerSize,u/l.batchSize,l.sliceSize];if(n.shouldExecuteOnCPU([i,a])||i.dtype===`string`){let e=n.bufferSync(m),t=AO(n.bufferSync(p),e,h);return d.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(l.outputShape,t.dtype,t.values)}let _=new iP(p.shape,h),v=n.runWebGLProgram(_,[p,m],p.dtype);d.push(v);let y=Q({inputs:{x:v},backend:n,attrs:{shape:l.outputShape}});return d.forEach(e=>n.disposeIntermediateTensorInfo(e)),y}var sP={kernelName:Dt,backendName:`webgl`,kernelFunc:oP},cP={kernelName:kt,backendName:`webgl`,kernelFunc:pA({opSnippet:`return float(a > b);`,packedOpSnippet:`
  return vec4(greaterThan(a, b));
`,cpuKernelImpl:jO,dtype:`bool`})},lP={kernelName:At,backendName:`webgl`,kernelFunc:pA({opSnippet:`return float(a >= b);`,packedOpSnippet:`
  return vec4(greaterThanEqual(a, b));
`,dtype:`bool`,cpuKernelImpl:MO})};function uP(e){let{inputs:t,backend:n}=e,{input:r}=t;return PN(r,!0,n)}var dP={kernelName:Mt,backendName:`webgl`,kernelFunc:uP},fP={kernelName:Pt,backendName:`webgl`,kernelFunc:Z({opSnippet:`return float(!isnan(x) && !isinf(x));`,dtype:`bool`})},pP={kernelName:Ft,backendName:`webgl`,kernelFunc:Z({opSnippet:`return float(isinf(x));`,dtype:`bool`})},mP={kernelName:It,backendName:`webgl`,kernelFunc:Z({opSnippet:`return float(isnan(x));`,dtype:`bool`})},hP={kernelName:Rt,backendName:`webgl`,kernelFunc:pA({opSnippet:`return float(a < b);`,packedOpSnippet:`
  return vec4(lessThan(a, b));
`,cpuKernelImpl:NO,dtype:`bool`})},gP={kernelName:zt,backendName:`webgl`,kernelFunc:pA({opSnippet:`return float(a <= b);`,packedOpSnippet:`
  return vec4(lessThanEqual(a, b));
`,cpuKernelImpl:PO,dtype:`bool`})};function _P(e){let{backend:t,attrs:n}=e,{start:r,stop:i,num:a}=n,o=FO(r,i,a);return t.makeTensorInfo([o.length],`float32`,o)}var vP={kernelName:Bt,backendName:`webgl`,kernelFunc:_P},yP={kernelName:`Log`,backendName:`webgl`,kernelFunc:Z({opSnippet:fA+`
  return x < 0.0 ? 0./0. : log(x);
`,packedOpSnippet:`
  vec4 result = log(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : (x.r < 0.0 ? 0./0. : result.r);
  result.g = isNaN.g ? x.g : (x.g < 0.0 ? 0./0. : result.g);
  result.b = isNaN.b ? x.b : (x.b < 0.0 ? 0./0. : result.b);
  result.a = isNaN.a ? x.a : (x.a < 0.0 ? 0./0. : result.a);
  return result;
`,cpuKernelImpl:IO})},bP={kernelName:Vt,backendName:`webgl`,kernelFunc:Z({opSnippet:fA+`
  return log(1.0 + x);
`})},xP={kernelName:Ht,backendName:`webgl`,kernelFunc:pA({opSnippet:`return float(a >= 1.0 && b >= 1.0);`,packedOpSnippet:`
  return vec4(
    vec4(greaterThanEqual(a, vec4(1.0))) *
    vec4(greaterThanEqual(b, vec4(1.0))));
`,dtype:`bool`})},SP={kernelName:Ut,backendName:`webgl`,kernelFunc:Z({opSnippet:`return float(!(x >= 1.0));`})},CP={kernelName:Wt,backendName:`webgl`,kernelFunc:pA({opSnippet:`return float(a >= 1.0 || b >= 1.0);`,packedOpSnippet:`
  return min(
    vec4(greaterThanEqual(a, vec4(1.0))) +
    vec4(greaterThanEqual(b, vec4(1.0))),
    vec4(1.0));
`,dtype:`bool`})},wP=class{constructor(e,t,n,r,i){this.variableNames=[`x`],this.outputShape=[];let a=t,o=e[3]-1;this.outputShape=e;let s,c=`float(${n}) + float(${r}) * sum`;s=i===.5?`inversesqrt(${c})`:i===1?`1.0/(${c})`:`exp(log(${c}) * float(-${i}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];
        int d = coords[3];
        float x = getX(b, r, c, d);
        float sum = 0.0;
        for (int j = -${a}; j <= ${a}; j++) {
          int idx = d + j;
          if (idx >= 0 && idx <=  ${o}) {
            float z = getX(b, r, c, idx);
            sum += z * z;
          }
        }
        float val = x * ${s};
        setOutput(val);
      }
    `}},TP=class{constructor(e,t,n,r,i){this.variableNames=[`x`],this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0;let a=t,o=e[3]-1;this.outputShape=e;let s,c=`float(${n}) + float(${r}) * sum`;s=i===.5?`inversesqrt(${c})`:i===1?`1.0/(${c})`:`exp(log(${c}) * float(-${i}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords.x;
        int r = coords.y;
        int c = coords.z;
        int d = coords.w;

        bool hasNextCol = d < ${this.outputShape[3]};
        bool hasNextRow = c < ${this.outputShape[2]};

        vec4 sum = vec4(0.);
        vec4 xFragAtOutputCoords = getX(b, r, c, d);

        vec4 xAtOutputCoords = vec4(
          getChannel(xFragAtOutputCoords, vec2(c, d)),
          hasNextCol ?
            getChannel(xFragAtOutputCoords, vec2(c, d + 1)) : 0.0,
          hasNextRow ?
            getChannel(xFragAtOutputCoords , vec2(c + 1, d)) : 0.0,
          (hasNextRow && hasNextCol) ?
            getChannel(xFragAtOutputCoords, vec2(c + 1, d + 1)) : 0.0
        );

        int firstChannel = d - ${a};
        vec2 cache = vec2(0.);
        if(firstChannel >= 0){
          vec4 firstChannelFrag = getX(b, r, c, firstChannel);
          cache.x = getChannel(firstChannelFrag, vec2(c, firstChannel));
            if(hasNextRow){
              cache.y = getChannel(firstChannelFrag, vec2(c + 1, firstChannel));
            }
        }

        ivec2 depth = ivec2(d, d + 1);
        for (int j = - ${a}; j <= ${a}; j++) {
          ivec2 idx = depth + j;
          bvec2 aboveLowerBound = greaterThanEqual(idx, ivec2(0));
          bvec2 belowUpperBound = lessThanEqual(idx, ivec2(${o}));

          bool depthInRange = aboveLowerBound.x && belowUpperBound.x;
          bool depthPlusOneInRange = aboveLowerBound.y && belowUpperBound.y;

          if(depthInRange || depthPlusOneInRange){
            vec4 z = vec4(0.);
            vec4 xFragAtCurrentDepth;
            z.xz = cache.xy;
            if(depthPlusOneInRange && hasNextCol){
              xFragAtCurrentDepth = idx.y != d ?
                getX(b, r, c, idx.y) : xFragAtOutputCoords;
              z.y = getChannel(xFragAtCurrentDepth, vec2(c, idx.y));
              if(hasNextRow){
                z.w = getChannel(xFragAtCurrentDepth, vec2(c + 1, idx.y));
              }
            }
            cache.xy = z.yw;
            sum += z * z;
          }
        }
        vec4 result = xAtOutputCoords * ${s};
        setOutput(result);
      }
    `}},EP={kernelName:`LRN`,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{depthRadius:a,bias:o,alpha:s,beta:c}=r,l=N().getBool(`WEBGL_PACK_NORMALIZATION`)?new TP(i.shape,a,o,s,c):new wP(i.shape,a,o,s,c);return n.runWebGLProgram(l,[i],i.dtype)}},DP=class{constructor(e,t,n,r,i){this.variableNames=[`inputImage`,`outputImage`,`dy`],this.outputShape=[],this.outputShape=e,this.depth=e[3],this.depthRadius=t,this.bias=n,this.alpha=r,this.beta=i,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];

        float result = 0.0;
        for (int d = 0; d < ${this.depth}; ++d) {
          int depthBegin = int(max(0.0, float(d - ${t})));
          int depthEnd = int(min(float(${this.depth}),
              float(d + ${t} + 1)));

          const int MIN_DEPTH_BEGIN = 0;
          const int MAX_DEPTH_END = ${this.depth};

          float norm = 0.0;
          for (int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k) {
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd) {
              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);
            }
            else {
              break;
            }
          }

          norm = float(${r}) * norm + float(${n});

          for(int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k){
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd){
              float dyi = -2.0 * float(${r})
                * float(${i})
                * getInputImage(b ,r ,c, k) * getOutputImage(b, r, c, d)
                / norm;
              if (k == d) {
                dyi += pow(norm, -1.0 * ${i});
              }
              if (k == coords[3]) {
                dyi *= getDy(b, r, c, d);
                result += dyi;
              }
            }
            else {
              break;
            }
          }
      }
      setOutput(result);
      }
    `}},OP={kernelName:Gt,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i,y:a,dy:o}=t,{depthRadius:s,bias:c,alpha:l,beta:u}=r,d=new DP(i.shape,s,c,l,u);return n.runWebGLProgram(d,[i,a,o],i.dtype)}};function kP(e,t,n,r){let i=g(t),a=g(e.shape)/i,o=Q({inputs:{x:e},attrs:{shape:[a,i]},backend:r}),s=EA(o,e.dtype,`max`,r),c=Q({inputs:{x:s},attrs:{shape:n},backend:r});return r.disposeIntermediateTensorInfo(o),r.disposeIntermediateTensorInfo(s),c}function AP(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{reductionIndices:a,keepDims:o}=r,s=i.shape.length,c=C(a,i.shape),l=c,u=_u(l,s),d=u!=null,f=n.shouldExecuteOnCPU([i]),p=i;if(d){if(f){let e=n.texData.get(p.dataId).values,t=Array(s);for(let e=0;e<t.length;e++)t[e]=i.shape[u[e]];let r=lk(e,i.shape,i.dtype,u,t);p=n.makeTensorInfo(t,i.dtype);let a=n.texData.get(p.dataId);a.values=r}else p=AA(i,u,n);l=yu(l.length,s)}gu(`max`,l,s);let[m,h]=mu(p.shape,l),_=m;o&&(_=hu(m,c));let v;if(f){let e=n.texData.get(p.dataId).values,t=LO(e,g(h),_,i.dtype);v=n.makeTensorInfo(_,i.dtype);let r=n.texData.get(v.dataId);r.values=t}else v=kP(p,h,_,n);return d&&n.disposeIntermediateTensorInfo(p),v}var jP={kernelName:`Max`,backendName:`webgl`,kernelFunc:AP},MP={kernelName:Kt,backendName:`webgl`,kernelFunc:pA({opSnippet:Xk+`
  return max(a, b);
`,packedOpSnippet:`
  vec4 result = vec4(max(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Qk+`
  return result;
`,cpuKernelImpl:RO})};function NP(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t;LE(i,`maxPool`);let{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=r;f(Tc(o,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${o} and dilations '1'`);let l=dc(i.shape,a,o,1,s,c);if(l.filterWidth===1&&l.filterHeight===1&&_(l.inShape,l.outShape))return eA({inputs:{x:i},backend:n});let u=new mj(l,`max`,!1);return n.runWebGLProgram(u,[i],i.dtype)}var PP={kernelName:qt,backendName:`webgl`,kernelFunc:NP};function FP(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dataFormat:c,dimRoundingMode:l}=r,u=new hj(fc(i.shape,a,o,[1,1,1],s,l,c),`max`,!1);return n.runWebGLProgram(u,[i],i.dtype)}var IP={kernelName:Yt,backendName:`webgl`,kernelFunc:FP},LP=class{constructor(e){this.variableNames=[`dy`,`maxPos`],this.outputShape=e.inShape;let t=e.strideHeight,n=e.strideWidth,r=e.dilationHeight,i=e.effectiveFilterHeight,a=e.effectiveFilterWidth,o=i-1-e.padInfo.top,s=a-1-e.padInfo.left,c=i*a-1;this.userCode=`
      const ivec2 pads = ivec2(${o}, ${s});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${i};
          wR += ${r}) {
          float dyR = float(dyRCorner + wR) / ${t}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${a}; wC++) {
            float dyC = float(dyCCorner + wC) / ${n}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);
            int maxPosValue = ${c} - int(getMaxPos(b, idyR, idyC, d));

            // Get the current value, check it against the value from the
            // position matrix.
            int curPosValue = wR * ${a} + wC;
            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

            dotProd += dyValue * mask;
          }
        }
        setOutput(dotProd);
      }
    `}},RP=class{constructor(e){this.variableNames=[`dy`,`maxPos`],this.outputShape=e.inShape;let t=e.strideDepth,n=e.strideHeight,r=e.strideWidth,i=e.dilationDepth,a=e.dilationHeight,o=e.dilationWidth,s=e.effectiveFilterDepth,c=e.effectiveFilterHeight,l=e.effectiveFilterWidth,u=s-1-e.padInfo.front,d=c-1-e.padInfo.top,f=l-1-e.padInfo.left,p=s*c*l-1;this.userCode=`
      const ivec3 pads = ivec3(${u}, ${d}, ${f});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${s};
           wD += ${i}) {
          float dyD = float(dyDCorner + wD) / ${t}.0;

          if (dyD < 0.0 || dyD >= ${e.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${c};
              wR += ${a}) {
            float dyR = float(dyRCorner + wR) / ${n}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${l};
                wC += ${o}) {
              float dyC = float(dyCCorner + wC) / ${r}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);
              int maxPosValue = ${p} -
                  int(getMaxPos(batch, idyD, idyR, idyC, ch));

              // Get the current value, check it against the value from the
              // position matrix.
              int curPosValue =
                  wD * ${c} * ${l} +
                  wR * ${l} + wC;
              float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

              dotProd += dyValue * mask;
            }
          }
        }
        setOutput(dotProd);
      }
    `}};function zP(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a,{filterSize:s,strides:c,pad:l,dimRoundingMode:u}=r,d=fc(o.shape,s,c,[1,1,1],l,u),f=new hj(d,`max`,!0),p=n.runWebGLProgram(f,[o],o.dtype),m=new RP(d),h=n.runWebGLProgram(m,[i,p],o.dtype);return n.disposeIntermediateTensorInfo(p),h}var BP={kernelName:Xt,backendName:`webgl`,kernelFunc:zP};function VP(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a,output:o}=t,s=a;LE([a,o],`maxPoolGrad`);let{filterSize:c,strides:l,pad:u,dimRoundingMode:d}=r,f=dc(s.shape,c,l,1,u,d),p=new mj(f,`max`,!0),m=n.runWebGLProgram(p,[s],s.dtype),h=new LP(f),g=n.runWebGLProgram(h,[i,m],s.dtype);return n.disposeIntermediateTensorInfo(m),g}var HP={kernelName:Jt,backendName:`webgl`,kernelFunc:VP};function UP(e,t,n,r){let i=new mj(n,`max`,!1),a=r.runWebGLProgram(i,[e],`float32`);return i=new mj(n,`max`,!0,!0,t),[a,r.runWebGLProgram(i,[e],`float32`)]}var WP={kernelName:Zt,backendName:`webgl`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{x:r}=e,{filterSize:i,strides:a,pad:o,includeBatchInIndex:s}=t,c=n;f(r.shape.length===4,()=>`Error in maxPool: input must be rank 4 but got rank ${r.shape.length}.`);let l=[1,1];f(Tc(a,l),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${a} and dilations '${l}'`);let[u,d]=UP(r,s,dc(r.shape,i,a,l,o),c);return[u,d]}};function GP(e,t,n,r){let i=g(t),a=g(e.shape)/i,o=Q({inputs:{x:e},attrs:{shape:[a,i]},backend:r}),s=EA(o,`float32`,`mean`,r),c=Q({inputs:{x:s},attrs:{shape:n},backend:r});return r.disposeIntermediateTensorInfo(o),r.disposeIntermediateTensorInfo(s),c}var KP={kernelName:Qt,backendName:`webgl`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{x:r}=e,{keepDims:i,axis:a}=t,o=n,s=r.shape.length,c=C(a,r.shape),l=c,u=_u(l,s),d=u!=null,f=o.shouldExecuteOnCPU([r]),p=[],m=r;if(d){if(f){let e=o.texData.get(m.dataId).values,t=Array(s);for(let e=0;e<t.length;e++)t[e]=r.shape[u[e]];let n=lk(e,r.shape,r.dtype,u,t);m=o.makeTensorInfo(t,r.dtype);let i=o.texData.get(m.dataId);i.values=n}else m=AA(r,u,o);p.push(m),l=yu(l.length,s)}gu(`sum`,l,s);let[h,g]=mu(m.shape,l),_=h;i&&(_=hu(h,c));let v=GP(m,g,_,o);for(let e of p)o.disposeIntermediateTensorInfo(e);return v}};function qP(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=i.shape.length,c=C(a,i.shape),l=c,u=_u(l,s),d=i;u!=null&&(d=PA({inputs:{x:i},backend:n,attrs:{perm:u}}),l=yu(l.length,i.shape.length)),gu(`min`,l,s);let[f,p]=mu(d.shape,l),m=g(p),h=Q({inputs:{x:d},backend:n,attrs:{shape:[-1,m]}}),_=EA(h,h.dtype,`min`,n),v;if(o){let e=hu(f,c);v=Q({inputs:{x:_},backend:n,attrs:{shape:e}})}else v=Q({inputs:{x:_},backend:n,attrs:{shape:f}});return n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(_),u!=null&&n.disposeIntermediateTensorInfo(d),v}var JP={kernelName:`Min`,backendName:`webgl`,kernelFunc:qP},YP={kernelName:$t,backendName:`webgl`,kernelFunc:pA({opSnippet:Xk+`
  return min(a, b);
`,packedOpSnippet:`
  vec4 result = vec4(min(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Qk+`
  return result;
`,cpuKernelImpl:zO})},XP=class{constructor(e,t,n){this.variableNames=[`x`],this.outputShape=t.map((t,n)=>t[0]+e[n]+t[1]);let r=e.length,i=X(r),a=t.map(e=>e[0]).join(`,`),o=t.map((t,n)=>t[0]+e[n]).join(`,`),s=[`coords[0]`,`coords[1]`,`coords[2]`,`coords[3]`].slice(0,r),c=n===`reflect`?0:1;this.userCode=r===1?`
        int start = ${a};
        int end = ${o};

        void main() {
          int outC = getOutputCoords();
          if (outC < start) {
            outC = start * 2 - outC - ${c};
          } else if(outC >= end) {
            outC = (end - 1) * 2 - outC + ${c};
          }
          setOutput(getX(outC - start));
        }
      `:`
      ${i} start = ${i}(${a});
      ${i} end = ${i}(${o});

      void main() {
        ${i} outC = getOutputCoords();
        for (int i = 0; i < ${r}; i++) {
          if (outC[i] < start[i]) {
            outC[i] = start[i] * 2 - outC[i] - ${c};
          } else if(outC[i] >= end[i]) {
            outC[i] = (end[i] - 1) * 2 - outC[i] + ${c};
          }
        }
        ${i} coords = outC - start;
        setOutput(getX(${s}));
      }
    `}},ZP=class{constructor(e,t,n){this.variableNames=[`x`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t.map((t,n)=>t[0]+e[n]+t[1]);let r=e.length,i=X(r),a=t.map(e=>e[0]).join(`,`),o=t.map((t,n)=>t[0]+e[n]).join(`,`),s=fk(`rc`,r),c=fk(`source`,r),l=`${s[r-1]} < ${this.outputShape[r-1]}`,u=r===1?`source`:`vec2(${c.slice(-2).join()})`,d=n===`reflect`?0:1,f=``;if(r===1){let e=`
        ${i} source = rc;
        if (source < start) {
          source = start * 2 - source - ${d};
        } else if (source >= end) {
          source = (end - 1) * 2 - source + ${d};
        }
        source -= start;
      `;f=`
        ${i} rc = outputLoc;
        ${e}
        result[0] = getChannel(getX(${c.join()}), ${u});
        ${s[r-1]} += 1;
        if(${l}) {
          ${e}
          result[1] = getChannel(getX(${c.join()}), ${u});
        }
      `}else{let e=`
        ${i} source = rc;
        ${i} lt = ${i}(lessThan(source, start));
        ${i} gte = ${i}(greaterThanEqual(source, end));
        ${i} orig = 1 - (lt + gte);
        source = orig * source +
                lt * (start * 2 - source - ${d}) +
                gte * ((end - 1) * 2 - source + ${d});
        source -= start;
      `;f=`
        ${i} rc = outputLoc;
        ${e}
        result[0] = getChannel(getX(${c.join()}), ${u});
        ${s[r-1]} += 1;
        if(${l}) {
          ${e}
          result[1] = getChannel(getX(${c.join()}), ${u});
        }
        rc = outputLoc;
        ${s[r-2]} += 1;
        if(${s[r-2]} < ${this.outputShape[r-2]}) {
          ${e}
          result[2] = getChannel(getX(${c.join()}), ${u});
          ${s[r-1]} += 1;
          if(${l}) {
            ${e}
            result[3] = getChannel(getX(${c.join()}), ${u});
          }
        }
      `}this.userCode=`
      const ${i} start = ${i}(${a});
      const ${i} end = ${i}(${o});

      void main() {
        ${i} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${f}
        setOutput(result);
      }
    `}},QP={kernelName:en,backendName:`webgl`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{let{x:r}=e,{paddings:i,mode:a}=n,o=N().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new ZP(r.shape,i,a):new XP(r.shape,i,a);return t.runWebGLProgram(o,[r],r.dtype)}},$P={kernelName:`Mod`,backendName:`webgl`,kernelFunc:pA({opSnippet:`if (b == 0.0) return NAN;
  return mod(a, b);`,packedOpSnippet:`
  vec4 result = mod(a, b);
  bvec4 isNaN = equal(b, vec4(0.0));
  `+Qk+`
  return result;
`})},eF=class{constructor(e,t,n){this.variableNames=[`probs`],this.customUniforms=[{name:`seed`,type:`float`}],this.outputShape=[e,n],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];

        float r = random(seed);
        float cdf = 0.0;

        for (int i = 0; i < ${t-1}; i++) {
          cdf += getProbs(batch, i);

          if (r < cdf) {
            setOutput(float(i));
            return;
          }
        }

        // If no other event happened, last event happened.
        setOutput(float(${t-1}));
      }
    `}},tF=pA({opSnippet:`
if (a == b) {
  return 1.0;
};
return a / b;`,packedOpSnippet:`
  // vec4 one = vec4(equal(a, b));
  // return one + (vec4(1.0) - one) * a / b;
  vec4 result = a / b;
  if(a.x == b.x) {
    result.x = 1.;
  }
  if(a.y == b.y) {
    result.y = 1.;
  }
  if(a.z == b.z) {
    result.z = 1.;
  }
  if(a.w == b.w) {
    result.w = 1.;
  }

  return result;
`,checkOutOfBounds:!0}),nF={kernelName:gt,backendName:`webgl`,kernelFunc:tF},rF=`return a - b;`,iF=pA({opSnippet:rF,packedOpSnippet:rF,supportsComplex:!0,cpuKernelImpl:ok}),aF={kernelName:`Sub`,backendName:`webgl`,kernelFunc:iF};function oF(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{dim:a}=r,o=C([a],i.shape),s=AP({inputs:{x:i},backend:n,attrs:{reductionIndices:o,keepDims:!1}}),c=hu(s.shape,o),l=Q({inputs:{x:s},backend:n,attrs:{shape:c}}),u=iF({inputs:{a:i,b:l},backend:n}),d=DN({inputs:{x:u},backend:n}),f=MA({inputs:{x:d},backend:n,attrs:{axis:o,keepDims:!1}}),p=Q({inputs:{x:f},backend:n,attrs:{shape:c}}),m=tF({inputs:{a:d,b:p},backend:n});return n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(l),n.disposeIntermediateTensorInfo(u),n.disposeIntermediateTensorInfo(d),n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(p),m}var sF={kernelName:Vn,backendName:`webgl`,kernelFunc:oF};function cF(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{numSamples:a,seed:o,normalized:s}=r,c=s?i:oF({inputs:{logits:i},backend:n,attrs:{dim:i.shape.length-1}}),l=c.shape[0],u=c.shape[1],d=new eF(l,u,a),f=[[o]],p=n.runWebGLProgram(d,[c],`int32`,f);return s||n.disposeIntermediateTensorInfo(c),p}var lF={kernelName:tn,backendName:`webgl`,kernelFunc:cF},uF=Tk+`
  return -x;
`,dF=`
  vec4 result = -x;
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`;function fF(e){let{inputs:t,backend:n}=e,{x:r}=t;if(n.shouldExecuteOnCPU([r])){let[e,t]=VO(n.texData.get(r.dataId).values,r.shape,r.dtype);return n.makeTensorInfo(t,r.dtype,e)}let i;return i=N().getBool(`WEBGL_PACK_UNARY_OPERATIONS`)?new Rk(r.shape,dF):new wk(r.shape,uF),n.runWebGLProgram(i,[r],r.dtype)}var pF={kernelName:`Neg`,backendName:`webgl`,kernelFunc:fF},mF=Kh;function hF(e){mr(`tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead`);let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c}=r,{selectedIndices:l}=mF(n.readSync(i.dataId),n.readSync(a.dataId),o,s,c);return n.makeTensorInfo([l.length],`int32`,new Int32Array(l))}var gF={kernelName:an,backendName:`webgl`,kernelFunc:hF},_F=qh;function vF(e){mr(`tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead`);let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c,padToMaxOutputSize:l}=r,{selectedIndices:u,validOutputs:d}=_F(n.readSync(i.dataId),n.readSync(a.dataId),o,s,c,l);return[n.makeTensorInfo([u.length],`int32`,new Int32Array(u)),n.makeTensorInfo([],`int32`,new Int32Array([d]))]}var yF={kernelName:on,backendName:`webgl`,kernelFunc:vF},bF=Jh;function xF(e){mr(`tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead`);let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c,softNmsSigma:l}=r,{selectedIndices:u,selectedScores:d}=bF(n.readSync(i.dataId),n.readSync(a.dataId),o,s,c,l);return[n.makeTensorInfo([u.length],`int32`,new Int32Array(u)),n.makeTensorInfo([d.length],`float32`,new Float32Array(d))]}var SF={kernelName:sn,backendName:`webgl`,kernelFunc:xF},CF=class{constructor(e,t,n,r){this.variableNames=[`indices`],this.outputShape=[e,t],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int index = round(getIndices(coords.x));
        setOutput(mix(float(${r}), float(${n}),
                      float(index == coords.y)));
      }
    `}},wF={kernelName:ln,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{indices:i}=t,{dtype:a,depth:o,onValue:s,offValue:c}=r,l=g(i.shape),u=new CF(l,o,s,c),d=Q({inputs:{x:i},backend:n,attrs:{shape:[l]}}),f=n.runWebGLProgram(u,[d],a);n.disposeIntermediateTensorInfo(d);let p=[...i.shape,o],m=Q({inputs:{x:f},backend:n,attrs:{shape:p}});return n.disposeIntermediateTensorInfo(f),m}};function TF(e){let{inputs:t,backend:n}=e,{x:r}=t;if(r.dtype===`complex64`){let e=Gj({inputs:{input:r},backend:n}),t=TF({inputs:{x:e},backend:n}),i=uM({inputs:{input:r},backend:n}),a=TF({inputs:{x:i},backend:n}),o=nA({inputs:{real:t,imag:a},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(a),o}return RN({attrs:{shape:r.shape,dtype:r.dtype,value:r.dtype===`string`?``:0},backend:n})}var EF={kernelName:sr,backendName:`webgl`,kernelFunc:TF};function DF(e){let{inputs:t,backend:n}=e,{x:r}=t;if(r.dtype===`string`)throw Error(`onesLike is not supported under string dtype`);if(r.dtype===`complex64`){let e=Gj({inputs:{input:r},backend:n}),t=DF({inputs:{x:e},backend:n}),i=uM({inputs:{input:r},backend:n}),a=TF({inputs:{x:i},backend:n}),o=nA({inputs:{real:t,imag:a},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(a),o}return RN({attrs:{shape:r.shape,dtype:r.dtype,value:1},backend:n})}var OF={kernelName:cn,backendName:`webgl`,kernelFunc:DF};function kF(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r;if(t.length===1)return kN({inputs:{input:t[0]},backend:n,attrs:{dim:i}});let a=t[0].shape,o=t[0].dtype;t.forEach(e=>{p(a,e.shape,`All tensors passed to stack must have matching shapes`),f(o===e.dtype,()=>`All tensors passed to stack must have matching dtypes`)});let s=[],c=mM({inputs:t.map(e=>{let t=kN({inputs:{input:e},backend:n,attrs:{dim:i}});return s.push(t),t}),backend:n,attrs:{axis:i}});return s.forEach(e=>n.disposeIntermediateTensorInfo(e)),c}var AF={kernelName:un,backendName:`webgl`,kernelFunc:kF},jF=class{constructor(e,t,n){this.variableNames=[`x`],this.customUniforms=[{name:`value`,type:`float`}],this.outputShape=t.map((t,n)=>t[0]+e[n]+t[1]);let r=e.length,i=X(r),a=t.map(e=>e[0]).join(`,`),o=t.map((t,n)=>t[0]+e[n]).join(`,`),s=[`coords[0]`,`coords[1]`,`coords[2]`,`coords[3]`].slice(0,r);this.userCode=r===1?`
        int start = ${a};
        int end = ${o};

        void main() {
          int outC = getOutputCoords();
          if (outC < start || outC >= end) {
            setOutput(value);
          } else {
            setOutput(getX(outC - start));
          }
        }
      `:`
      ${i} start = ${i}(${a});
      ${i} end = ${i}(${o});

      void main() {
        ${i} outC = getOutputCoords();
        if (any(lessThan(outC, start)) || any(greaterThanEqual(outC, end))) {
          setOutput(value);
        } else {
          ${i} coords = outC - start;
          setOutput(getX(${s}));
        }
      }
    `}},MF=class{constructor(e,t,n){this.variableNames=[`x`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`value`,type:`float`}],this.outputShape=t.map((t,n)=>t[0]+e[n]+t[1]);let r=e.length,i=X(r),a=t.map(e=>e[0]).join(`,`),o=t.map((t,n)=>t[0]+e[n]).join(`,`),s=fk(`rc`,r),c=fk(`source`,r),l=`${s[r-1]} < ${this.outputShape[r-1]}`,u=r===1?`source`:`vec2(${c.slice(-2).join()})`,d=[`${i} rc = outputLoc;`,`${s[r-1]} += 1;
       if(${l}) {
      `,r===1?``:`}
       rc = outputLoc;
       ${s[r-2]} += 1;
       if(${s[r-2]} < ${this.outputShape[r-2]}) {`,r===1?``:`  ${s[r-1]} += 1;
         if(${l}) {`],f=r===1?`rc < start || rc >= end`:`any(lessThan(rc, start)) || any(greaterThanEqual(rc, end))`,p=``;for(let e=0,t=r===1?2:4;e<t;e++)p+=`
        ${d[e]}
        if (${f}) {
          result[${e}] = float(value);
        } else {
          ${i} source = rc - start;
          result[${e}] = getChannel(getX(${c.join()}), ${u});
        }
      `;p+=r===1?`} `:`}}`,this.userCode=`
      const ${i} start = ${i}(${a});
      const ${i} end = ${i}(${o});

      void main() {
        ${i} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${p}
        setOutput(result);
      }
    `}},NF=e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{paddings:a,constantValue:o}=r;if(g(i.shape)===0)return RN({backend:n,attrs:{shape:a.map((e,t)=>e[0]+i.shape[t]+e[1]),value:o,dtype:i.dtype}});let s=N().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new MF(i.shape,a,o):new jF(i.shape,a,o),c=[[o]];return n.runWebGLProgram(s,[i],i.dtype,c)},PF={kernelName:dn,backendName:`webgl`,kernelFunc:NF},FF={kernelName:`Pow`,backendName:`webgl`,kernelFunc:pA({opSnippet:`
  if(a < 0.0 && floor(b) < b){
    return NAN;
  }
  if (b == 0.0) {
    return 1.0;
  }
  return (round(mod(b, 2.0)) != 1) ?
      pow(abs(a), b) : sign(a) * pow(abs(a), b);
`,packedOpSnippet:`
  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));
  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);
  vec4 result = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  bvec4 isExpZero = equal(b, vec4(0.0));
  result.r = isExpZero.r ? 1.0 : result.r;
  result.g = isExpZero.g ? 1.0 : result.g;
  result.b = isExpZero.b ? 1.0 : result.b;
  result.a = isExpZero.a ? 1.0 : result.a;

  bvec4 isNaN1 = lessThan(a, vec4(0.0));
  bvec4 isNaN2 = lessThan(floor(b), b);
  bvec4 isNaN = bvec4(isNaN1.x && isNaN2.x, isNaN1.y && isNaN2.y, isNaN1.z && isNaN2.z, isNaN1.w && isNaN2.w);
  `+Qk+`
  return result;
`})};function IF(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=i.shape.length,c=[],l=C(a,i.shape),u=l,d=_u(u,s),f=i;d!=null&&(f=PA({inputs:{x:i},backend:n,attrs:{perm:d}}),u=yu(u.length,s),c.push(f)),gu(`prod`,u,s);let p;if(n.shouldExecuteOnCPU([f])){let e=n.texData.get(f.dataId).values,{outVals:t,outShape:r,outDtype:i}=UO(f.shape,f.dtype,e,u);p=n.makeTensorInfo(r,i,t)}else{let[e,t]=mu(f.shape,u),r=g(t),a=Q({inputs:{x:f},backend:n,attrs:{shape:[-1,r]}}),o=EA(a,Si(i.dtype),`prod`,n);p=Q({inputs:{x:o},backend:n,attrs:{shape:e}}),c.push(a),c.push(o)}if(o){c.push(p);let e=hu(p.shape,l);p=Q({inputs:{x:p},backend:n,attrs:{shape:e}})}return c.forEach(e=>n.disposeIntermediateTensorInfo(e)),p}var LF={kernelName:pn,backendName:`webgl`,kernelFunc:IF};function RF(e){let{inputs:t,backend:n,attrs:r}=e,{paramsNestedSplits:i,paramsDenseValues:a,indices:o}=t,{outputRaggedRank:s}=r,c=i.map(e=>n.readSync(e.dataId)),l=i.map(e=>e.shape),u=n.readSync(a.dataId),d=n.readSync(o.dataId),[f,p,m]=WO(c,l,u,a.shape,a.dtype,d,o.shape,s),h=f.map(e=>n.makeTensorInfo([e.length],`int32`,e)),g=n.makeTensorInfo(m,a.dtype,p);return h.concat([g])}var zF={kernelName:mn,backendName:`webgl`,kernelFunc:RF};function BF(e){let{inputs:t,backend:n,attrs:r}=e,{shape:i,values:a,defaultValue:o,rowPartitionTensors:s}=t,{rowPartitionTypes:c}=r,l=n.readSync(i.dataId),u=n.readSync(a.dataId),d=n.readSync(o.dataId),f=s.map(e=>n.readSync(e.dataId)),p=s.map(e=>e.shape),[m,h]=GO(l,i.shape,u,a.shape,a.dtype,d,o.shape,f,p,c);return n.makeTensorInfo(m,a.dtype,h)}var VF={kernelName:hn,backendName:`webgl`,kernelFunc:BF},HF=e=>{let{backend:t,attrs:n}=e,{start:r,stop:i,step:a,dtype:o}=n,s=KO(r,i,a,o);return t.makeTensorInfo([s.length],o,s)},UF={kernelName:gn,backendName:`webgl`,kernelFunc:HF},WF={kernelName:vn,backendName:`webgl`,kernelFunc:Z({opSnippet:`return 1.0 / x;`})},GF={kernelName:yn,backendName:`webgl`,kernelFunc:Z({opSnippet:Tk+`
  return (x < 0.0) ? 0.0 : x;
`,packedOpSnippet:`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`})},KF={kernelName:Tn,backendName:`webgl`,kernelFunc:Z({opSnippet:Tk+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,packedOpSnippet:`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`})},qF=class{constructor(e,t,n,r,i){this.variableNames=[`A`],this.outputShape=[];let[a,o,s,c]=e;this.outputShape=[a,t,n,c];let l=[r&&t>1?o-1:o,r&&n>1?s-1:s],u=[r&&t>1?t-1:t,r&&n>1?n-1:n],d;d=i?`(vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC - vec2(0.5)`:`vec2(yRC) * effectiveInputOverOutputRatioRC`,this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${l[0]/u[0]},
          ${l[1]/u[1]});
      const vec2 inputShapeRC = vec2(${o}.0, ${s}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${d};

        // Compute the four integer indices.
        ivec2 sourceFloorRC = ivec2(max(sourceFracIndexRC, vec2(0.0)));
        ivec2 sourceCeilRC = ivec2(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        float topLeft = getA(b, sourceFloorRC.x, sourceFloorRC.y, d);
        float bottomLeft = getA(b, sourceCeilRC.x, sourceFloorRC.y, d);
        float topRight = getA(b, sourceFloorRC.x, sourceCeilRC.y, d);
        float bottomRight = getA(b, sourceCeilRC.x, sourceCeilRC.y, d);

        vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

        float top = topLeft + (topRight - topLeft) * fracRC.y;
        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
        float newValue = top + (bottom - top) * fracRC.x;

        setOutput(newValue);
      }
    `}},JF=class{constructor(e,t,n,r,i){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];let[a,o,s,c]=e;this.outputShape=[a,t,n,c];let l=[r&&t>1?o-1:o,r&&n>1?s-1:s],u=[r&&t>1?t-1:t,r&&n>1?n-1:n],d;d=i?`(vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC - vec3(0.5)`:`vec3(yRC) * effectiveInputOverOutputRatioRC`,this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${l[0]/u[0]},
          ${l[1]/u[1]},
          ${l[1]/u[1]});
      const vec3 inputShapeRC = vec3(${o}.0, ${s}.0,
                                     ${s}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${d};

        // Compute the four integer indices.
        ivec3 sourceFloorRC = ivec3(max(sourceFracIndexRC, vec3(0.0)));
        ivec3 sourceCeilRC = ivec3(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${c-1};
        bool hasNextRow = coords.z < ${n-1};

        // In parallel, construct four corners for all four components in
        // packed 2x2 cell.
        vec4 topLeft = vec4(
          getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 bottomLeft = vec4(
          getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 topRight = vec4(
          getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec4 bottomRight = vec4(
          getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec3 fracRC = sourceFracIndexRC - vec3(sourceFloorRC);

        vec4 top = mix(topLeft, topRight, fracRC.yyzz);
        vec4 bottom = mix(bottomLeft, bottomRight, fracRC.yyzz);
        vec4 newValue = mix(top, bottom, fracRC.x);

        setOutput(newValue);
      }
    `}};function YF(e){let{inputs:t,backend:n,attrs:r}=e,{images:i}=t,{alignCorners:a,halfPixelCenters:o,size:s}=r,[c,l]=s,u=N().getBool(`WEBGL_PACK_IMAGE_OPERATIONS`)?new JF(i.shape,c,l,a,o):new qF(i.shape,c,l,a,o);return n.runWebGLProgram(u,[i],`float32`)}var XF={kernelName:Cn,backendName:`webgl`,kernelFunc:YF},ZF=class{constructor(e,t,n){this.variableNames=[`dy`],this.outputShape=[],this.outputShape=t;let[,r,i]=t,[,a,o]=e,s=[n&&a>1?r-1:r,n&&o>1?i-1:i],c=[n&&a>1?a-1:a,n&&o>1?o-1:o],l=s[0]/c[0],u=s[1]/c[1],d=1/l,f=1/u,p=Math.ceil(d)*2+2,m=Math.ceil(f)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${l});
        const float widthScale = float(${u});

        const float invHeightScale = float(${d});
        const float invWidthScale = float(${f});

        const int winHeight = int(${p});
        const int winWidth = int(${m});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(startRLerp - float(winHeight / 2));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(startCLerp - float(winWidth / 2));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${a}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${o}) {
              continue;
            }

            float dxR = float(dyR) * heightScale;
            int topDxRIndex = int(floor(dxR));
            int bottomDxRIndex = int(min(ceil(dxR), ${r-1}.0));
            float dxRLerp = dxR - float(topDxRIndex);
            float inverseDxRLerp = 1.0 - dxRLerp;

            float dxC = float(dyC) * widthScale;
            int leftDxCIndex = int(floor(dxC));
            int rightDxCIndex = int(min(ceil(dxC), ${i-1}.0));
            float dxCLerp = dxC - float(leftDxCIndex);
            float inverseDxCLerp = 1.0 - dxCLerp;

            if (r == topDxRIndex && c == leftDxCIndex) {
              // topLeft
              accumulator +=
                getDy(b, dyR, dyC, d) * inverseDxRLerp * inverseDxCLerp;
            }

            if (r == topDxRIndex && c == rightDxCIndex) {
              // topRight
              accumulator += getDy(b, dyR, dyC, d) * inverseDxRLerp * dxCLerp;
            }

            if (r == bottomDxRIndex && c == leftDxCIndex) {
              // bottomLeft
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * inverseDxCLerp;
            }

            if (r == bottomDxRIndex && c == rightDxCIndex) {
              // bottomRight
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * dxCLerp;
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}};function QF(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r,s=new ZF(a.shape,i.shape,o);return n.runWebGLProgram(s,[a],a.dtype)}var $F={kernelName:wn,backendName:`webgl`,kernelFunc:QF},eI=class{constructor(e,t,n,r,i){this.variableNames=[`A`],this.outputShape=[];let[a,o,s,c]=e;this.outputShape=[a,t,n,c];let l=[r&&t>1?o-1:o,r&&n>1?s-1:s],u=[r&&t>1?t-1:t,r&&n>1?n-1:n],d=r?`0.5`:`0.0`,f;f=i?`max((vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC, vec2(0.0))`:`vec2(yRC) * effectiveInputOverOutputRatioRC`,this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${l[0]/u[0]},
          ${l[1]/u[1]});
      const vec2 inputShapeRC = vec2(${o}.0, ${s}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${f};

        // Compute the coordinators of nearest neighbor point.
        ivec2 sourceNearestRC = ivec2(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${d})));
        float newValue = getA(b, sourceNearestRC.x, sourceNearestRC.y, d);

        setOutput(newValue);
      }
    `}},tI=class{constructor(e,t,n,r,i){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];let[a,o,s,c]=e;this.outputShape=[a,t,n,c];let l=[r&&t>1?o-1:o,r&&n>1?s-1:s],u=[r&&t>1?t-1:t,r&&n>1?n-1:n],d=r?`0.5`:`0.0`,f;f=i?`max((vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC, vec3(0.0))`:`vec3(yRC) * effectiveInputOverOutputRatioRC`,this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${l[0]/u[0]},
          ${l[1]/u[1]},
          ${l[1]/u[1]});
      const vec3 inputShapeRC = vec3(${o}.0, ${s}.0,
                                     ${s}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${f};

        // Compute the coordinators of nearest neighbor point.
        ivec3 sourceNearestRC = ivec3(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${d})));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${c-1};
        bool hasNextRow = coords.z < ${n-1};

        vec4 newValue = vec4(
          getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d),
          hasNextCol ? getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d + 1) : 0.0);

        setOutput(newValue);
      }
    `}};function nI(e){let{inputs:t,backend:n,attrs:r}=e,{images:i}=t,{alignCorners:a,halfPixelCenters:o,size:s}=r,[c,l]=s,u=N().getBool(`WEBGL_PACK_IMAGE_OPERATIONS`)?new tI(i.shape,c,l,a,o):new eI(i.shape,c,l,a,o);return n.runWebGLProgram(u,[i],i.dtype)}var rI={kernelName:xn,backendName:`webgl`,kernelFunc:nI},iI=class{constructor(e,t,n){this.variableNames=[`dy`],this.outputShape=[],this.outputShape=t;let[,r,i]=t,[,a,o]=e,s=[n&&a>1?r-1:r,n&&o>1?i-1:i],c=[n&&a>1?a-1:a,n&&o>1?o-1:o],l=s[0]/c[0],u=s[1]/c[1],d=1/l,f=1/u,p=Math.ceil(d)*2+2,m=Math.ceil(f)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${l});
        const float widthScale = float(${u});

        const float invHeightScale = float(${d});
        const float invWidthScale = float(${f});

        const int winHeight = int(${p});
        const int winWidth = int(${m});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(floor(startRLerp - float(winHeight / 2)));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(floor(startCLerp - float(winWidth / 2)));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${a}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${o}) {
              continue;
            }

            float sourceFracRow =
              float(${s[0]}) *
                (float(dyR) / float(${c[0]}));

            float sourceFracCol =
                float(${s[1]}) *
                  (float(dyC) / float(${c[1]}));

            int sourceNearestRow = int(min(
                float(int(${r}) - 1),
                ${n} ? float(round(sourceFracRow)) :
                                  float(floor(sourceFracRow))));

            int sourceNearestCol = int(min(
                float(int(${i}) - 1),
                ${n} ? float(round(sourceFracCol)) :
                                  float(floor(sourceFracCol))));

            if (r == sourceNearestRow && c == sourceNearestCol) {
              accumulator += getDy(b, dyR, dyC, d);
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}};function aI(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r,s=new iI(a.shape,i.shape,o);return n.runWebGLProgram(s,[a],a.dtype)}var oI={kernelName:Sn,backendName:`webgl`,kernelFunc:aI},sI=class{constructor(e,t){this.variableNames=[`x`];let n=e.length;if(n>4)throw Error(`WebGL backend: Reverse of rank-${n} tensor is not yet supported`);if(this.outputShape=e,n===1){this.userCode=`
        void main() {
          int coord = getOutputCoords();
          setOutput(getX(${e[0]} - coord - 1));
        }
      `;return}let r=n=>t.indexOf(n)!==-1&&e[n]!==1?`${e[n]} - coords[${n}] - 1`:`coords[${n}]`,i=e.map((e,t)=>r(t)).join(`,`),a=X(n);this.userCode=`
      void main() {
        ${a} coords = getOutputCoords();
        setOutput(getX(${i}));
      }
    `}},cI=class{constructor(e,t){this.variableNames=[`x`],this.packedInputs=!0,this.packedOutput=!0;let n=e.length;if(n>4)throw Error(`WebGL backend: Reverse of rank-${n} tensor is not yet supported`);this.outputShape=e;let r=fk(`rc`,n),i=`${r[n-1]} + 1 < ${this.outputShape[n-1]}`,a=`${r[n-2]} + 1 < ${this.outputShape[n-2]}`,o=X(n);this.userCode=n===1?`
        void main(){
          int rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = getChannel(getX(${e[0]} - rc - 1),
            ${e[0]} - rc - 1);
          if(${i}){
              result.g = getChannel(getX(${e[0]} - (rc  + 1) - 1),
                ${e[0]} - (rc  + 1) - 1);
          }
          setOutput(result);
        }
      `:`
        void main() {
          ${o} rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = ${s(r.slice())};
          if(${i}){
            result.g = ${c(r.slice())};
          }
          if(${a}) {
            result.b = ${l(r.slice())};
            if(${i}) {
              result.a = ${u(r.slice())};
            }
          }
          setOutput(result);
        }
    `;function s(e){return d(e)}function c(e){return e[n-1]=`(`+e[n-1]+` + 1)`,d(e)}function l(e){return e[n-2]=`(`+e[n-2]+` + 1)`,d(e)}function u(e){return e[n-1]=`(`+e[n-1]+` + 1)`,e[n-2]=`(`+e[n-2]+` + 1)`,d(e)}function d(t){let n=e.map((e,n)=>f(n,t));return`getChannel(getX(${n.join(`,`)}), vec2(${n.slice(-2).join(`,`)}))`}function f(n,r){return t.indexOf(n)!==-1&&e[n]!==1?`${e[n]} - ${r[n]} - 1`:`${r[n]}`}}};function lI(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dims:a}=r,o=i.shape.length,s=C(a,i.shape);if(o===0)return eA({inputs:{x:i},backend:n});let c=N().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new cI(i.shape,s):new sI(i.shape,s);return n.runWebGLProgram(c,[i],i.dtype)}var uI={kernelName:En,backendName:`webgl`,kernelFunc:lI},dI=class{constructor(e,t){this.variableNames=[`Image`],this.outputShape=[],this.customUniforms=[{name:`params`,type:`vec4`}];let n=e[1],r=e[2];this.outputShape=e;let i=``;i=typeof t==`number`?`float outputValue = ${t.toFixed(2)};`:`
        vec3 fill = vec3(${t.join(`,`)});
        float outputValue = fill[coords[3]];`,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];
          int y = coords[1];
          float coordXFloat = (float(x) - params[0]) * params[3] -
            (float(y) - params[1]) * params[2];
          float coordYFloat = (float(x) - params[0]) * params[2] +
            (float(y) - params[1]) * params[3];
          int coordX = int(round(coordXFloat + params[0]));
          int coordY = int(round(coordYFloat + params[1]));
          ${i}
          if(coordX >= 0 && coordX < ${r} && coordY >= 0 && coordY < ${n}) {
            outputValue = getImage(coords[0], coordY, coordX, coords[3]);
          }
          setOutput(outputValue);
        }
    `}},fI={kernelName:ur,backendName:`webgl`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{image:r}=e,{radians:i,fillValue:a,center:o}=t,s=n,c=new dI(r.shape,a),[l,u]=C_(o,r.shape[1],r.shape[2]),d=[[l,u,Math.sin(i),Math.cos(i)]];return s.runWebGLProgram(c,[r],r.dtype,d)}},pI={kernelName:Dn,backendName:`webgl`,kernelFunc:Z({opSnippet:`
  // OpenGL ES does not support round function.
  // The algorithm is based on banker's rounding.
  float base = floor(x);
  if ((x - base) < 0.5) {
    return floor(x);
  } else if ((x - base) > 0.5) {
    return ceil(x);
  } else {
    if (mod(base, 2.0) == 0.0) {
      return base;
    } else {
      return base + 1.0;
    }
  }
`})},mI={kernelName:On,backendName:`webgl`,kernelFunc:Z({opSnippet:`return inversesqrt(x);`,cpuKernelImpl:qO})},hI=class{constructor(e,t,n,r,i,a,o=!0){this.variableNames=[`updates`,`indices`,`defaultValue`],this.outputShape=a;let s=X(i.length),c=X(a.length),l=``;n===1?l=`i`:n===2&&(l=`i, j`);let u=`getIndices(${l})`,d=``;r===1?d=`i`:r===2&&(d=`i, coords[1]`);let f=`getUpdates(${d})`,p=t>1?`strides[j]`:`strides`;this.userCode=`
        ${s} strides = ${s}(${i});

        void main() {
          ${c} coords = getOutputCoords();
          float sum = 0.0;
          bool found = false;
          for (int i = 0; i < ${e}; i++) {
            int flattenedIndex = 0;
            for (int j = 0; j < ${t}; j++) {
              int index = round(${u});
              flattenedIndex += index * ${p};
            }
            if (flattenedIndex == coords[0]) {
              sum += ${f};
              found = true;
            }
          }
          setOutput(mix(getDefaultValue(), sum, float(found)));
        }
      `}};function gI(e){let{inputs:t,backend:n,attrs:r}=e,{indices:i,updates:a}=t,{shape:o}=r,{sliceRank:s,numUpdates:c,sliceSize:l,strides:u,outputSize:d}=ls(a,i,o),f=[d/l,l];if(d===0)return n.makeTensorInfo(o,i.dtype);let p=Q({inputs:{x:i},backend:n,attrs:{shape:[c,s]}}),m=Q({inputs:{x:a},backend:n,attrs:{shape:[c,l]}}),h=n.makeTensorInfo([],`float32`,new Float32Array([0])),g=new hI(c,s,p.shape.length,m.shape.length,u,f),_=n.runWebGLProgram(g,[m,p,h],m.dtype),v=Q({inputs:{x:_},backend:n,attrs:{shape:o}});return n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(_),n.disposeIntermediateTensorInfo(h),v}var _I={kernelName:kn,backendName:`webgl`,kernelFunc:gI},vI=class{constructor(e,t,n,r){this.variableNames=[`sortedSequence`,`values`],this.customUniforms=[{name:`numInputs`,type:`int`}],this.outputShape=[e,n];let i=`for (int i = 0; i < ${Math.ceil(Math.log2(t+1))}; ++i) { if (left >= right) break;`,a=N().getNumber(`WEBGL_VERSION`)===2?`while (left < right) {`:i,o=r===`left`?`<`:`<=`;this.userCode=`
       int findBound(int batch, float value) {
         int left = 0;
         int right = numInputs;
         int mid;
         ${a}
           mid = (left + right) / 2;
           if (getSortedSequence(batch, mid) ${o} value) {
             left = mid + 1;
           } else {
             right = mid;
           }
         }
         return right;
       }

       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int valueIndex = coords[1];

         float value = getValues(batch, valueIndex);

         setOutput(float(findBound(batch, value)));
       }
     `}};function yI(e){let{inputs:t,backend:n,attrs:r}=e,{sortedSequence:i,values:a}=t,{side:o}=r,s=new vI(i.shape[0],i.shape[1],a.shape[1],o),c=[[i.shape[1]]];return n.runWebGLProgram(s,[i,a],`int32`,c)}var bI={kernelName:An,backendName:`webgl`,kernelFunc:yI},xI=class{constructor(e,t,n){this.variableNames=[`c`,`a`,`b`],this.outputShape=t;let r,i;if(n>4)throw Error(`Where for rank ${n} is not yet supported`);if(n===1)i=`resRC`,r=`resRC`;else{let n=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`],a=[],o=[];for(let r=0;r<t.length;r++)o.push(`${n[r]}`),r<e&&a.push(`${n[r]}`);r=a.join(),i=o.join()}let a=X(n);this.userCode=`
      void main() {
        ${a} resRC = getOutputCoords();
        float cVal = getC(${r});
        if (cVal >= 1.0) {
          setOutput(getA(${i}));
        } else {
          setOutput(getB(${i}));
        }
      }
    `}};function SI(e){let{inputs:t,backend:n}=e,{condition:r,t:i,e:a}=t,o=new xI(r.shape.length,i.shape,i.shape.length);return n.runWebGLProgram(o,[r,i,a],xi(i.dtype,a.dtype))}var CI={kernelName:jn,backendName:`webgl`,kernelFunc:SI},wI={kernelName:Mn,backendName:`webgl`,kernelFunc:Z({opSnippet:`
  // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
  // see: https://arxiv.org/abs/1706.02515
  float scaleAlpha = ${k_};
  float scale = ${A_};
  return (x >= 0.0) ? scale * x : scaleAlpha * (exp(x) - 1.0);
`})},TI={kernelName:In,backendName:`webgl`,kernelFunc:Z({opSnippet:fA+`
  return 1.0 / (1.0 + exp(-1.0 * x));
`,packedOpSnippet:`
  vec4 result = 1.0 / (1.0 + exp(-1.0 * x));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,cpuKernelImpl:YO})},EI={kernelName:Fn,backendName:`webgl`,kernelFunc:Z({opSnippet:`
  if (isnan(x)) { return 0.0; }
  return sign(x);
`})},DI={kernelName:`Sin`,backendName:`webgl`,kernelFunc:Z({opSnippet:fA+`
  return sin(x);
`})},OI={kernelName:Pn,backendName:`webgl`,kernelFunc:Z({opSnippet:`
  float e2x = exp(x);
  return (e2x - 1.0 / e2x) / 2.0;
`})},kI={kernelName:Ln,backendName:`webgl`,kernelFunc:Z({opSnippet:`
  float epsilon = 1.1920928955078125e-7;
  float threshold = log(epsilon) + 2.0;

  bool too_large = x > -threshold;
  bool too_small = x < threshold;

  float result;
  float exp_x = exp(x);

  if (too_large){
    result = x;
  }
  else if (too_small){
    result = exp_x;
  }
  else{
    result = log(exp_x + 1.0);
  }
  return result;
`})},AI={kernelName:zn,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,paddings:o}=r;f(i.shape.length<=4,()=>`spaceToBatchND for rank > 4 with a WebGL backend not implemented yet`);let s=a.reduce((e,t)=>e*t),c=[[0,0]];c.push(...o);for(let e=1+a.length;e<i.shape.length;++e)c.push([0,0]);let l=[],u=NF({inputs:{x:i},backend:n,attrs:{paddings:c,constantValue:0}}),d=w_(u.shape,a,s,!1),p=T_(d.length,a.length,!1),m=E_(u.shape,a,s,!1),h=Q({inputs:{x:u},backend:n,attrs:{shape:d}}),g=PA({inputs:{x:h},backend:n,attrs:{perm:p}}),_=Q({inputs:{x:g},backend:n,attrs:{shape:m}});return l.push(u),l.push(h),l.push(g),l.forEach(e=>n.disposeIntermediateTensorInfo(e)),_}};function jI(e){let{inputs:t,backend:n}=e,{indices:r,values:i,denseShape:a,defaultValue:o}=t;if(a.shape.length!==1)throw Error(`Dense shape must be a vector, saw:
         ${a.shape}`);if(r.shape.length!==2)throw Error(`Indices must be a matrix, saw:
         ${r.shape}`);if(i.shape.length!==1)throw Error(`Values must be a vector, saw:
         ${i.shape}`);if(o.shape.length!==0)throw Error(`Default value must be a scalar, saw:
        ${o.shape}`);let s=n.readSync(r.dataId),c=n.readSync(i.dataId),l=n.readSync(a.dataId),u=n.readSync(o.dataId)[0],[d,f,p,m,h]=QO(s,r.shape,r.dtype,c,i.dtype,l,u);return[n.makeTensorInfo(f,r.dtype,d),n.makeTensorInfo([f[0]],i.dtype,p),n.makeTensorInfo([m.length],`bool`,new Uint8Array(m.map(e=>Number(e)))),n.makeTensorInfo([h.length],r.dtype,new Int32Array(h))]}var MI={kernelName:Hn,backendName:`webgl`,kernelFunc:jI};function NI(e){let{inputs:t,backend:n}=e,{inputIndices:r,inputShape:i,newShape:a}=t;if(r.shape.length!==2)throw Error(`Input indices should be a matrix but received shape ${r.shape}`);if(i.shape.length!==1)throw Error(`Input shape should be a vector but received shape ${i.shape}`);if(a.shape.length!==1)throw Error(`Target shape should be a vector but received shape ${a.shape}`);let o=Array.from(n.readSync(i.dataId)),s=n.readSync(r.dataId),c=Array.from(n.readSync(a.dataId)),[l,u,d]=$O(s,r.shape,r.dtype,o,c);return[n.makeTensorInfo(u,r.dtype,l),n.makeTensorInfo([d.length],a.dtype,new Int32Array(d))]}var PI={kernelName:Un,backendName:`webgl`,kernelFunc:NI};function FI(e){let{inputs:t,backend:n}=e,{data:r,indices:i,segmentIds:a}=t;if(r.shape.length<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.shape.length!==1)throw Error(`Indices should be a vector but received shape
              ${i.shape}`);if(a.shape.length!==1)throw Error(`Segment ids should be a vector but received shape
              ${a.shape}`);let o=n.readSync(r.dataId),s=n.readSync(i.dataId),c=n.readSync(a.dataId),[l,u]=ek(o,r.shape,r.dtype,s,c,!0);return n.makeTensorInfo(u,r.dtype,l)}var II={kernelName:Wn,backendName:`webgl`,kernelFunc:FI};function LI(e){let{inputs:t,backend:n}=e,{data:r,indices:i,segmentIds:a}=t;if(r.shape.length<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.shape.length!==1)throw Error(`Indices should be a vector but received shape
             ${i.shape}`);if(a.shape.length!==1)throw Error(`Segment ids should be a vector but received shape
             ${a.shape}`);let o=n.readSync(r.dataId),s=n.readSync(i.dataId),c=n.readSync(a.dataId),[l,u]=ek(o,r.shape,r.dtype,s,c);return n.makeTensorInfo(u,r.dtype,l)}var RI={kernelName:Gn,backendName:`webgl`,kernelFunc:LI};function zI(e){let{inputs:t,backend:n,attrs:r}=e,{sparseIndices:i,sparseValues:a,defaultValue:o}=t,{outputShape:s}=r,{sliceRank:c,numUpdates:l,sliceSize:u,strides:d,outputSize:f}=ls(a,i,s);if(a.dtype===`string`){let e=JO(n.bufferSync(i),n.bufferSync(a),s,f,u,l,c,d,Kr(n.readSync(o.dataId)[0]),!1);return n.makeTensorInfo(s,e.dtype,e.values)}let p=new hI(l,c,i.shape.length,a.shape.length,d,[f,1],!1),m=n.runWebGLProgram(p,[a,i,o],a.dtype),h=Q({inputs:{x:m},backend:n,attrs:{shape:s}});return n.disposeIntermediateTensorInfo(m),h}var BI={kernelName:Kn,backendName:`webgl`,kernelFunc:zI};function VI(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{numOrSizeSplits:a,axis:o}=r,s=C(o,i.shape)[0],c=tv(i,a,s),l=i.shape.length,u=Array(l).fill(0),d=i.shape.slice();return c.map(e=>{let t=[...d];t[s]=e;let r=Ij({inputs:{x:i},backend:n,attrs:{begin:u,size:t}});return u[s]+=e,r})}var HI={kernelName:Bn,backendName:`webgl`,kernelFunc:VI},UI=`return sqrt(x);`,WI={kernelName:Rn,backendName:`webgl`,kernelFunc:Z({opSnippet:UI,packedOpSnippet:UI,cpuKernelImpl:tk})},GI={kernelName:Jn,backendName:`webgl`,kernelFunc:Z({opSnippet:`return x * x;`})},KI=`return (a - b) * (a - b);`,qI={kernelName:qn,backendName:`webgl`,kernelFunc:pA({opSnippet:KI,packedOpSnippet:KI})};function JI({inputs:e,attrs:t,backend:n}){let{x:r}=e,i=Tk+`
    return x > 0.0 ? 1.0 : float(${t.alpha});
  `,a=new wk(r.shape,i);return n.runWebGLProgram(a,[r],r.dtype)}var YI={kernelName:cr,backendName:`webgl`,kernelFunc:JI},XI=class{constructor(e,t,n){this.variableNames=[`x`],this.outputShape=n;let r=n.length,i=X(n.length),a=X(n.length),o=``;if(r===1)o=`coords * strides + begin`;else{let e=0;o=n.map((t,r)=>(e++,n.length===1?`coords * strides[${r}] + begin[${r}]`:`coords[${e-1}] * strides[${r}] + begin[${r}]`)).join(`,`)}this.userCode=`
      ${i} begin = ${i}(${e});
      ${i} strides = ${i}(${t});

      void main() {
        ${a} coords = getOutputCoords();
        setOutput(getX(${o}));
      }
    `}};function ZI(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{begin:a,end:o,strides:s,beginMask:c,endMask:l,ellipsisMask:u,newAxisMask:d,shrinkAxisMask:p}=r,{finalShapeSparse:m,finalShape:h,isIdentity:g,sliceDim0:_,isSimpleSlice:v,begin:y,end:b,strides:x}=Os(i.shape,a,o,s,c,l,u,d,p),S;if(g)S=Q({inputs:{x:i},backend:n,attrs:{shape:h}});else if(_||v){f(i.shape.length>=1,()=>`Input must have rank at least 1, got: ${i.shape.length}`);let e=hs(y,b,x),t=Ij({inputs:{x:i},backend:n,attrs:{begin:y,size:e}});S=Q({inputs:{x:t},backend:n,attrs:{shape:h}}),n.disposeIntermediateTensorInfo(t)}else if(n.shouldExecuteOnCPU([i])){let e=n.readSync(i.dataId),t=nk(m,L(i.shape,i.dtype,e),x,y);S=n.makeTensorInfo(h,i.dtype,t.values)}else{let e=new XI(y,x,m);S=n.runWebGLProgram(e,[i],i.dtype)}let C=Q({inputs:{x:S},backend:n,attrs:{shape:h}});return n.disposeIntermediateTensorInfo(S),C}var QI={kernelName:Yn,backendName:`webgl`,kernelFunc:ZI};function $I(e){let{inputs:t,backend:n,attrs:r}=e,{separator:i,nGramWidths:a,leftPad:o,rightPad:s,padWidth:c,preserveShortSequences:l}=r,{data:u,dataSplits:d}=t,[f,p]=rk(n.readSync(u.dataId),n.readSync(d.dataId),i,a,o,s,c,l);return[n.makeTensorInfo([f.length],`string`,f),n.makeTensorInfo(d.shape,`int32`,p)]}var eL={kernelName:Xn,backendName:`webgl`,kernelFunc:$I};function tL(e){let{inputs:t,backend:n,attrs:r}=e,{skipEmpty:i}=r,{input:a,delimiter:o}=t;if(a.dtype!==`string`)throw Error(`Input must be of datatype string`);if(a.shape.length!==1)throw Error(`Input must be a vector, got shape: ${a.shape}`);if(o.shape.length!==0)throw Error(`Delimiter must be a scalar, got shape: ${o.shape}`);let s=n.readSync(a.dataId),c=n.readSync(o.dataId)[0],[l,u,d]=ik(s,c,i),f=u.length;return[n.makeTensorInfo([f,2],`int32`,l),n.makeTensorInfo([f],`string`,u),n.makeTensorInfo([2],`int32`,new Int32Array(d))]}var nL={kernelName:Zn,backendName:`webgl`,kernelFunc:tL};function rL(e){let{inputs:t,backend:n,attrs:r}=e,{numBuckets:i}=r,{input:a}=t;if(a.dtype!==`string`)throw Error(`Input must be of datatype string`);if(i<=0)throw Error(`Number of buckets must be at least 1`);let o=ak(n.readSync(a.dataId),i);return n.makeTensorInfo(a.shape,`int32`,o)}var iL={kernelName:Qn,backendName:`webgl`,kernelFunc:rL},aL={kernelName:`Tan`,backendName:`webgl`,kernelFunc:Z({opSnippet:`return tan(x);`})},oL={kernelName:$n,backendName:`webgl`,kernelFunc:Z({opSnippet:`
  float e2x = exp(-2.0 * abs(x));
  return sign(x) * (1.0 - e2x) / (1.0 + e2x);
`})},sL=class{constructor(e,t){this.variableNames=[`A`];let n=Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[r]*t[r];this.outputShape=n,this.rank=n.length;let r=X(this.rank),i=cL(e);this.userCode=`
      void main() {
        ${r} resRC = getOutputCoords();
        setOutput(getA(${i}));
      }
    `}};function cL(e){let t=e.length;if(t>5)throw Error(`Tile for rank ${t} is not yet supported`);if(t===1)return`imod(resRC, ${e[0]})`;let n=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`,`resRC.u`],r=[];for(let t=0;t<e.length;t++)r.push(`imod(${n[t]}, ${e[t]})`);return r.join()}function lL(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{reps:a}=r;if(i.dtype===`string`||i.shape.length>5){let e=n.readSync(i.dataId),t=i.dtype===`string`?e.map(e=>Kr(e)):e,r=sk(L(i.shape,i.dtype,t),a);return n.makeTensorInfo(r.shape,r.dtype,r.values)}let o=new sL(i.shape,a);return n.runWebGLProgram(o,[i],i.dtype)}var uL={kernelName:er,backendName:`webgl`,kernelFunc:lL},dL=class{constructor(e){this.variableNames=[`x`,`indices`],this.customUniforms=[{name:`n`,type:`int`},{name:`firstPass`,type:`int`},{name:`negativeInf`,type:`float`},{name:`dir`,type:`int`},{name:`inc`,type:`int`}],this.outputShape=e,this.userCode=`
       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // We compare elements pair-wise within a group of size 2 * inc.
         // The comparing rule for each group alternates between ascending
         // and descending. Within each group, we compare each pair at
         // positions i and i+inc. To decide whether an element at position i
         // is x0 or x1, we mod it by 2 * inc, if the result is smaller than
         // inc, it is in the first half of the group, we denote it as x0,
         // otherwise we denote it as x1.
         // For example, as shown in the Bitonic top K paper referenced above,
         // Figure5(a) shows that element[1] is in the
         // second half of the group when group size is 2, but it is in the
         // first half of the group when group size is 4.

         bool isFirstInPair = imod(elemIdx, 2 * inc) < inc;
         int i = isFirstInPair ? elemIdx : elemIdx - inc;

         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + inc : int(getIndices(batch, i + inc));
         float x0 = i0 < n ? getX(batch, i0) : negativeInf;
         float x1 = i1 < n ? getX(batch, i1) : negativeInf;

         // Denotes which direction indices are in (ascending or descending).
         bool reverse = imod(elemIdx, 2 * dir) >= dir;
         bool isGreater = x0 > x1 || (x0 == x1 && i1 > i0);
         if (reverse == isGreater) { // Elements in opposite order of direction
           int iTemp = i0;
           i0 = i1;
           i1 = iTemp;
         }
         if (isFirstInPair) {
            setOutput(float(i0));
         } else {
            setOutput(float(i1));
         }
       }
     `}},fL=class{constructor(e){this.variableNames=[`x`,`indices`],this.customUniforms=[{name:`n`,type:`int`},{name:`firstPass`,type:`int`},{name:`k`,type:`int`}],this.outputShape=e,this.userCode=`
    void main() {
         // Takes max of indices (0, k), (1, k + 1), (2, k + 2) ...
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // The output size is half of the previous size.
         // If the previous sequence is | | | | _ _ _ _  | | | |  _ _ _ _ (k=4),
         // we only need to output the indices at positions |, the indices at
         // positions _ can be thrown away, see Figure5(b) After Phase 2
         // (Merge phase) in the Bitonic Top K paper referenced above.
         // For example, the paper shows we only need to output the orange bars.
         // The output sequence should look like this | | | | | | | |.
         // Because the sequence is halved, to map the output index back
         // to the previous sequence to find the corresponding value,
         // we need to double the index. When we double the index,
         // we basically interpolate a position, so 2i looks like
         // | _ | _ | _ | _ | _ | _ | _. We move the | to the first k position
         // of each 2k positions by - elemIdx % k. E.g. for output at
         // index 4,5,6,7, we want to get the corresponding element at
         // original index 8,9,10,11, for output at index 8,9,10,11,
         // we want to get the corresponding element at original index
         // 16,17,18,19, so on and so forth.

         int i = elemIdx < k ? elemIdx : (elemIdx * 2 - imod(elemIdx, k));
         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + k : int(getIndices(batch, i + k));

         float x0 = getX(batch, i0);
         float x1 = i1 < n ? getX(batch, i1) : x0;

         setOutput(x0 >= x1 ? float(i0) : float(i1));
       }
     `}};function pL(e,t){t!==null&&e.disposeIntermediateTensorInfo(t)}function mL(e){let t=1;for(;t<e;)t*=2;return t}function hL(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{k:a,sorted:o}=r,s=N().getNumber(`TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD`),c=N().getNumber(`TOPK_K_CPU_HANDOFF_THRESHOLD`),l=i.shape,u=l[l.length-1];if(n.shouldExecuteOnCPU([i])||u<s||a>c){let[e,t]=ck(n.readSync(i.dataId),l,i.dtype,a,o);return[n.makeTensorInfo(e.shape,e.dtype,e.values),n.makeTensorInfo(t.shape,t.dtype,t.values)]}if(a===0)return l[l.length-1]=0,[n.makeTensorInfo(l,i.dtype,[]),n.makeTensorInfo(l,`int32`,[])];if(u===1)return[i,RN({attrs:{shape:l,dtype:`int32`,value:0},backend:n})];let d=n.texData.get(i.dataId),f=d!==null&&d.isPacked,p=f?n.unpackTensor(i):i,m=g(l)/u,h=Q({inputs:{x:p},attrs:{shape:[m,u]},backend:n});f&&pL(n,p);let _=mL(a),v=mL(u),y=null,b=()=>y===null?[h,h]:[h,y],x=(e,t,r)=>{let i=b(),a=new dL(r),o=[[u],[+(y===null)],[-1/0],[e],[t]],s=y;y=n.runWebGLProgram(a,i,`int32`,o),pL(n,s)};for(let e=1;e<_;e*=2){let t=e*2;for(let n=e;n>=1;n/=2)x(t,n,[m,v])}for(let e=v;e>_;e/=2){let t=b(),r=new fL([m,e/2]),i=[[u],[+(y===null)],[_]],a=y;y=n.runWebGLProgram(r,t,`int32`,i),pL(n,a);let o=_/2,s=o*2;for(let e=o;e>=1;e/=2)x(s,e,y.shape)}let S=y;y=Ij({inputs:{x:y},backend:n,attrs:{begin:0,size:[m,a]}}),pL(n,S);let C=oP({inputs:{x:h,indices:y},backend:n,attrs:{axis:1,batchDims:1}});pL(n,h);let w=l.slice(0,-1);w.push(a),S=y,y=Q({inputs:{x:y},attrs:{shape:w},backend:n}),pL(n,S);let T=C;return C=Q({inputs:{x:C},attrs:{shape:w},backend:n}),pL(n,T),[C,y]}var gL={kernelName:tr,backendName:`webgl`,kernelFunc:hL},_L=class{constructor(e,t,n,r,i,a){this.variableNames=[`Image`,`Transforms`],this.outputShape=a;let o=n===`nearest`?1:2,s;switch(r){case`constant`:s=1;break;case`reflect`:s=2;break;case`wrap`:s=3;break;case`nearest`:s=4;break;default:s=1}this.userCode=`
            float mapCoord(float outCoord, float len) {
              float inCoord = outCoord;
              if(${s} == 2) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    if (inCoord < sz2) {
                      inCoord = sz2 * float(int(float(-inCoord / sz2))) +
                      inCoord;
                    }
                    inCoord = inCoord < -len ? inCoord + sz2 : -inCoord - 1.0;
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    inCoord -= sz2 * float(int(float(inCoord / sz2)));
                    if (inCoord >= len) {
                      inCoord = sz2 - inCoord - 1.0;
                    }
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${s} == 3) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord += len * (float(int(float(-inCoord / sz))) + 1.0);
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord -= len * float(int(float(inCoord / sz)));
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${s} == 4) {
                return clamp(outCoord, 0.0, len - 1.0);
              } else {
                return outCoord;
              }
            }

            float readWithFillValue(int batch, int coordY, int coordX,
              int channel) {
              float outputValue;
              if (0 <= coordY && coordY < ${e} && 0 <= coordX && coordX < ${t}) {
                  outputValue = getImage(batch, coordY, coordX, channel);
              } else {
                outputValue = float(${i});
              }
              return outputValue;
            }

            void main() {
              ivec4 coords = getOutputCoords();
              float outputValue;
              int batch = coords[0];
              int x = coords[2];
              int y = coords[1];
              int channel = coords[3];
              float xf = float(x);
              float yf = float(y);
              float a1 = getTransforms(batch, 0);
              float a2 = getTransforms(batch, 1);
              float a3 = getTransforms(batch, 2);
              float b1 = getTransforms(batch, 3);
              float b2 = getTransforms(batch, 4);
              float b3 = getTransforms(batch, 5);
              float c1 = getTransforms(batch, 6);
              float c2 = getTransforms(batch, 7);
              float projection = c1 * xf + c2 * yf + 1.0;
              if (projection == 0.0) {
                outputValue = float(${i});
              } else {
                float inX = (a1 * xf + a2 * yf + a3) / projection;
                float inY = (b1 * xf + b2 * yf + b3) / projection;
                float mapX = mapCoord(inX, float(${t}));
                float mapY = mapCoord(inY, float(${e}));

                if (${o} == 1) {
                  int coordY = int(round(mapY));
                  int coordX = int(round(mapX));
                  outputValue = readWithFillValue(batch, coordY, coordX,
                    channel);
                } else {
                  float yFloor = floor(mapY);
                  float xFloor = floor(mapX);
                  float yCeil = yFloor + 1.0;
                  float xCeil = xFloor + 1.0;
                  float valueYFloor = (xCeil - mapX) *
                  readWithFillValue(batch, int(yFloor), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yFloor), int(xCeil), channel);
                  float valueYCeil = (xCeil - mapX) *
                  readWithFillValue(batch, int(yCeil), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yCeil), int(xCeil), channel);
                  outputValue = (yCeil - mapY) * valueYFloor +
                  (mapY - yFloor) * valueYCeil;
                }
              }
              setOutput(outputValue);
            }
        `}};function vL(e){let{inputs:t,backend:n,attrs:r}=e,{image:i,transforms:a}=t,{interpolation:o,fillMode:s,fillValue:c,outputShape:l}=r,[u,d,f,p]=i.shape,[m,h]=l??[d,f],g=new _L(d,f,o,s,c,[u,m,h,p]);return n.runWebGLProgram(g,[i,a],`float32`)}var yL={kernelName:nr,backendName:`webgl`,kernelFunc:vL};function bL(e){let{inputs:t,attrs:n,backend:r}=e,{axis:i}=n,{x:a}=t;LE(a,`unique`),console.warn(`WARNING: `,`UI might be locked temporarily as data is being downloaded`);let{outputValues:o,outputShape:s,indices:c}=uk(r.readSync(a.dataId),i,a.shape,a.dtype);return[r.makeTensorInfo(s,a.dtype,o),r.makeTensorInfo([c.length],`int32`,c)]}var xL={kernelName:ir,backendName:`webgl`,kernelFunc:bL};function SL(e){let{inputs:t,backend:n,attrs:r}=e,{value:i}=t,{axis:a}=r;a<0&&(a+=i.shape.length);let o=i,s=o.shape.length,c=i.shape[a],l=Array(s-1),u=0;for(let e=0;e<s;e++)e!==a&&(l[u++]=o.shape[e]);let d=[],f=Array(s).fill(0),p=o.shape.slice();p[a]=1;let m=Array(c);for(let e=0;e<m.length;e++){f[a]=e;let t=Ij({inputs:{x:o},backend:n,attrs:{begin:f,size:p}}),r=Q({inputs:{x:t},backend:n,attrs:{shape:l}});m[e]=r,d.push(t)}return d.forEach(e=>n.disposeIntermediateTensorInfo(e)),m}var CL={kernelName:ar,backendName:`webgl`,kernelFunc:SL},wL=class{constructor(e,t){this.variableNames=[`x`,`segmentIds`];let n=e.windowSize,r=e.batchSize,i=e.inSize,a=e.numSegments,o=a*Math.ceil(i/n);this.outputShape=[r,o];let s=Math.floor(n/4)*4,c=n%4,l=`
        sumValue += dot(values, segFilter);
    `,u=``;i%n>0&&(u=`
        if (inIdx < 0 || inIdx >= ${i}) {
          return initializationValue;
        }
      `);let d=``;i%n>0&&(d=`
        if (inIdx < 0 || inIdx >= ${i}) {
          return -1.0;
        }
      `),this.userCode=`
      const float initializationValue = 0.0;

      float getValue(int batch, int inIdx) {
        ${u}
        return getX(batch, inIdx);
      }

      float getSegmentIdAtIndex(int inIdx) {
        ${d}
        return getSegmentIds(inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = int(floor(float(outIdx) / float(
          ${a})) * float(${n}));
        int currentSeg = int(mod(float(outIdx), float(${a})));

        float sumValue = 0.0;

        for (int i = 0; i < ${s}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 3)) == currentSeg ? 1 : 0
          );

          ${l}
        }

        int inIdx = inOffset + ${s};
        if (${c===1}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          int inIdxSeg = int(getSegmentIdAtIndex(inIdx));

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            0,
            0,
            0
          );

          ${l}
        } else if (${c===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
              0,
              0
          );

          ${l}
        } else if (${c===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            0
          );

          ${l}
        }
        setOutput(sumValue);
      }
    `}};function TL(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,segmentIds:a}=t,{numSegments:o}=r,s=i.shape.length,c=[],l=0,u=_u([l],s),d=i;u!=null&&(d=PA({inputs:{x:i},backend:n,attrs:{perm:u}}),c.push(d),l=yu(1,s)[0]);let f=gv(d.shape,l,o),p=g([d.shape[l]]),m=Q({inputs:{x:d},backend:n,attrs:{shape:[-1,p]}});c.push(m);let h=Si(i.dtype),_=(e,t,r,i,a)=>{let o=e.shape[0],s=e.shape[1],l=hv(s,a),u=new wL({windowSize:l,inSize:s,batchSize:o,numSegments:a},t),d=n.compileAndRun(u,[e,r],i);if(c.push(d),d.shape[1]===a)return d;let f=HF({backend:n,attrs:{start:0,stop:a,step:1,dtype:`float32`}}),p=lL({inputs:{x:f},backend:n,attrs:{reps:[s/l]}});return c.push(f),c.push(p),_(d,t,p,i,a)},v=Q({inputs:{x:_(m,`unsortedSegmentSum`,a,h,o)},backend:n,attrs:{shape:f}}),y=v;if(u!=null){c.push(v);let e=vu(u);y=PA({inputs:{x:y},backend:n,attrs:{perm:e}})}return c.forEach(e=>n.disposeIntermediateTensorInfo(e)),y}var EL=[RA,VA,HA,UA,GA,YA,ZA,$A,oj,cj,lj,uj,dj,fj,pj,_j,yj,Cj,Tj,Dj,Aj,Rj,Bj,Hj,Xj,Qj,nM,rA,oM,hM,wM,AM,MM,PM,IM,RM,zM,BM,HM,YM,ZM,$M,nN,oN,uN,fN,hN,vN,bN,xN,wN,TN,EN,ON,AN,MN,IN,zN,VN,UN,WN,qN,QN,eP,rP,sP,cP,lP,tA,dP,dM,fP,pP,mP,sA,hP,gP,vP,yP,bP,xP,SP,CP,EP,OP,jP,MP,PP,IP,BP,HP,WP,KP,JP,YP,QP,$P,lF,bA,pF,gF,yF,SF,Wj,wF,OF,AF,PF,FF,dA,LF,zF,VF,UF,Kj,nF,WF,GF,KF,SA,XF,$F,rI,oI,uI,fI,pI,mI,_I,bI,CI,wI,TI,EI,DI,OI,Lj,sF,kI,AI,MI,PI,II,RI,BI,HI,WI,GI,qI,YI,QI,eL,nL,iL,aF,NA,aL,oL,uL,gL,yL,FA,xL,CL,{kernelName:or,backendName:`webgl`,kernelFunc:TL},EF];for(let e of EL)xr(e);N().registerFlag(`KEEP_INTERMEDIATE_TENSORS`,()=>!1,e=>{e&&console.warn(`Keep intermediate tensors is ON. This will print the values of all intermediate tensors during model inference. Not all models support this mode. For details, check e2e/benchmarks/ model_config.js. This significantly impacts performance.`)});var DL;(function(e){e[e.DT_INVALID=0]=`DT_INVALID`,e[e.DT_FLOAT=1]=`DT_FLOAT`,e[e.DT_DOUBLE=2]=`DT_DOUBLE`,e[e.DT_INT32=3]=`DT_INT32`,e[e.DT_UINT8=4]=`DT_UINT8`,e[e.DT_INT16=5]=`DT_INT16`,e[e.DT_INT8=6]=`DT_INT8`,e[e.DT_STRING=7]=`DT_STRING`,e[e.DT_COMPLEX64=8]=`DT_COMPLEX64`,e[e.DT_INT64=9]=`DT_INT64`,e[e.DT_BOOL=10]=`DT_BOOL`,e[e.DT_QINT8=11]=`DT_QINT8`,e[e.DT_QUINT8=12]=`DT_QUINT8`,e[e.DT_QINT32=13]=`DT_QINT32`,e[e.DT_BFLOAT16=14]=`DT_BFLOAT16`,e[e.DT_QINT16=15]=`DT_QINT16`,e[e.DT_QUINT16=16]=`DT_QUINT16`,e[e.DT_UINT16=17]=`DT_UINT16`,e[e.DT_COMPLEX128=18]=`DT_COMPLEX128`,e[e.DT_HALF=19]=`DT_HALF`,e[e.DT_RESOURCE=20]=`DT_RESOURCE`,e[e.DT_VARIANT=21]=`DT_VARIANT`,e[e.DT_UINT32=22]=`DT_UINT32`,e[e.DT_UINT64=23]=`DT_UINT64`,e[e.DT_FLOAT_REF=101]=`DT_FLOAT_REF`,e[e.DT_DOUBLE_REF=102]=`DT_DOUBLE_REF`,e[e.DT_INT32_REF=103]=`DT_INT32_REF`,e[e.DT_UINT8_REF=104]=`DT_UINT8_REF`,e[e.DT_INT16_REF=105]=`DT_INT16_REF`,e[e.DT_INT8_REF=106]=`DT_INT8_REF`,e[e.DT_STRING_REF=107]=`DT_STRING_REF`,e[e.DT_COMPLEX64_REF=108]=`DT_COMPLEX64_REF`,e[e.DT_INT64_REF=109]=`DT_INT64_REF`,e[e.DT_BOOL_REF=110]=`DT_BOOL_REF`,e[e.DT_QINT8_REF=111]=`DT_QINT8_REF`,e[e.DT_QUINT8_REF=112]=`DT_QUINT8_REF`,e[e.DT_QINT32_REF=113]=`DT_QINT32_REF`,e[e.DT_BFLOAT16_REF=114]=`DT_BFLOAT16_REF`,e[e.DT_QINT16_REF=115]=`DT_QINT16_REF`,e[e.DT_QUINT16_REF=116]=`DT_QUINT16_REF`,e[e.DT_UINT16_REF=117]=`DT_UINT16_REF`,e[e.DT_COMPLEX128_REF=118]=`DT_COMPLEX128_REF`,e[e.DT_HALF_REF=119]=`DT_HALF_REF`,e[e.DT_RESOURCE_REF=120]=`DT_RESOURCE_REF`,e[e.DT_VARIANT_REF=121]=`DT_VARIANT_REF`,e[e.DT_UINT32_REF=122]=`DT_UINT32_REF`,e[e.DT_UINT64_REF=123]=`DT_UINT64_REF`})(DL||={});var OL;(function(e){(function(e){e[e.LEGACY=0]=`LEGACY`,e[e.V1=1]=`V1`,e[e.V2=2]=`V2`})(e.CheckpointFormatVersion||={})})(OL||={});var kL={};function AL(e){return kL[e]}function $(e,t,n,r,i){let a=t.inputParams[e];if(a&&a.inputIndexStart!==void 0){let e=a.inputIndexStart,o=a.inputIndexEnd===0?void 0:a.inputIndexEnd===void 0?e+1:a.inputIndexEnd;if(a.type===`tensor`)return jL(t.inputNames[a.inputIndexStart],n,r,i);if(a.type===`tensors`)return t.inputNames.slice(e,o).map(e=>jL(e,n,r,i));let s=jL(t.inputNames.slice(e)[0],n,r,i),c=s.dataSync();return a.type===`number`?c[0]:le(s.shape,c)}let o=t.attrParams[e];return o&&o.value}function jL(e,t,n,r){let[i,a]=FL(e);if(r!=null){let e=r.getHashTableHandleByName(i);if(e!=null)return e}let o=n.currentContextIds.find(e=>!!t[PL(i,e)]);return o===void 0?void 0:t[PL(i,o)][a]}function ML(e,t,n){return t[PL(e,n.currentContextId)]}function NL(e,t){let[n,r,i]=FL(e);return[PL(n,t&&t.currentContextId),r,i]}function PL(e,t){return t?`${e}-${t}`:e}function FL(e){let t=e.split(`:`);if(t.length===1)return[e,0,void 0];let n=t[0],r=t.length===3?t[1]:void 0;return[n,Number(t[t.length-1]),r]}function IL(e,t,n){let r=$(`pad`,e,t,n);if(r===`explicit`){r=$(`explicitPaddings`,e,t,n);let i=[[0,0],[0,0],[0,0],[0,0]];for(let e=0;e<4;e++)i[e][0]=r[e*2],i[e][1]=r[e*2+1];return i}return r}function LL(e){return e.kept?e:oo(e)}var RL=t({json:()=>zL}),zL=[{tfOpName:`Add`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`AddV2`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`AddN`,category:`arithmetic`,inputs:[{start:0,end:0,name:`tensors`,type:`tensors`}]},{tfOpName:`BiasAdd`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0}]},{tfOpName:`Sub`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`RealDiv`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Div`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`DivNoNan`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`FloorDiv`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Mul`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Maximum`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Minimum`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Pow`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`SquaredDifference`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Mod`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`FloorMod`,category:`arithmetic`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]}],BL=t({json:()=>VL}),VL=[{tfOpName:`Abs`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Acos`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Asin`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Atan`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Atan2`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`y`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Ceil`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`ClipByValue`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`clipValueMin`,type:`number`},{start:2,name:`clipValueMax`,type:`number`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Complex`,category:`basic_math`,inputs:[{start:0,name:`real`,type:`tensor`},{start:1,name:`imag`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`ComplexAbs`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Cos`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Cosh`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Elu`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Exp`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Floor`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Log`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Imag`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`Tout`,name:`outputType`,type:`dtype`,notSupported:!0}]},{tfOpName:`Neg`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Real`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`Tout`,name:`outputType`,type:`dtype`,notSupported:!0}]},{tfOpName:`Prelu`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`alpha`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Relu`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Relu6`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Selu`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Sigmoid`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Sin`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Sinh`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Sqrt`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Rsqrt`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Square`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Tan`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Tanh`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Sign`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Round`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Expm1`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Log1p`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Reciprocal`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Softplus`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Asinh`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Acosh`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Atanh`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Erf`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Prod`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axes`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`,notSupported:!0},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`LeakyRelu`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`alpha`,name:`alpha`,type:`number`,defaultValue:.2},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`IsNan`,category:`basic_math`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]}],HL=t({json:()=>UL}),UL=[{tfOpName:`EmptyTensorList`,category:`control`,inputs:[{start:0,name:`elementShape`,type:`shape`},{start:1,name:`maxNumElements`,type:`number`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`LoopCond`,category:`control`,inputs:[{start:0,name:`pred`,type:`tensor`}]},{tfOpName:`Switch`,category:`control`,inputs:[{start:0,name:`data`,type:`tensor`},{start:1,name:`pred`,type:`tensor`}]},{tfOpName:`Merge`,category:`control`,inputs:[{start:0,end:0,name:`tensors`,type:`tensors`}]},{tfOpName:`Enter`,category:`control`,inputs:[{start:0,name:`tensor`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`frame_name`,name:`frameName`,type:`string`},{tfName:`is_constant`,name:`isConstant`,type:`bool`}]},{tfOpName:`Exit`,category:`control`,inputs:[{start:0,name:`tensor`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`NextIteration`,category:`control`,inputs:[{start:0,name:`tensor`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`TensorArrayV3`,category:`control`,inputs:[{start:0,name:`size`,type:`number`}],attrs:[{tfName:`dtype`,name:`dtype`,type:`dtype`},{tfName:`element_shape`,name:`elementShape`,type:`shape`},{tfName:`dynamic_size`,name:`dynamicSize`,type:`bool`},{tfName:`clear_after_read`,name:`clearAfterRead`,type:`bool`},{tfName:`identical_element_shapes`,name:`identicalElementShapes`,type:`bool`},{tfName:`tensor_array_name`,name:`name`,type:`string`}]},{tfOpName:`TensorArrayWriteV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`},{start:1,name:`index`,type:`number`},{start:2,name:`tensor`,type:`tensor`},{start:3,name:`flowIn`,type:`number`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`TensorArrayReadV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`},{start:1,name:`index`,type:`number`},{start:2,name:`flowIn`,type:`number`}],attrs:[{tfName:`dtype`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`TensorArrayGatherV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`},{start:1,name:`indices`,type:`number[]`},{start:2,name:`flowIn`,type:`number`}],attrs:[{tfName:`dtype`,name:`dtype`,type:`dtype`},{tfName:`element_shape`,name:`elementShape`,type:`shape`}]},{tfOpName:`TensorArrayScatterV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`},{start:1,name:`indices`,type:`number[]`},{start:2,name:`tensor`,type:`tensor`},{start:3,name:`flowIn`,type:`number`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`TensorArrayConcatV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`},{start:1,name:`flowIn`,type:`number`}],attrs:[{tfName:`dtype`,name:`dtype`,type:`dtype`},{tfName:`element_shape_except0`,name:`elementShapeExcept0`,type:`shape`,notSupported:!0}]},{tfOpName:`TensorArraySplitV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`},{start:1,name:`tensor`,type:`tensor`},{start:2,name:`lengths`,type:`number[]`},{start:3,name:`flowIn`,type:`number`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`TensorArraySizeV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`},{start:1,name:`flowIn`,type:`number`}]},{tfOpName:`TensorArrayCloseV3`,category:`control`,inputs:[{start:0,name:`tensorArrayId`,type:`tensor`}]},{tfOpName:`StatelessIf`,category:`control`,inputs:[{start:0,name:`cond`,type:`tensor`},{start:1,end:0,name:`args`,type:`tensors`}],attrs:[{tfName:`then_branch`,name:`thenBranch`,type:`func`},{tfName:`else_branch`,name:`elseBranch`,type:`func`}]},{tfOpName:`If`,category:`control`,inputs:[{start:0,name:`cond`,type:`tensor`},{start:1,end:0,name:`args`,type:`tensors`}],attrs:[{tfName:`then_branch`,name:`thenBranch`,type:`func`},{tfName:`else_branch`,name:`elseBranch`,type:`func`}]},{tfOpName:`StatelessWhile`,category:`control`,inputs:[{start:0,end:0,name:`args`,type:`tensors`}],attrs:[{tfName:`cond`,name:`cond`,type:`func`},{tfName:`body`,name:`body`,type:`func`}]},{tfOpName:`While`,category:`control`,inputs:[{start:0,end:0,name:`args`,type:`tensors`}],attrs:[{tfName:`cond`,name:`cond`,type:`func`},{tfName:`body`,name:`body`,type:`func`}]},{tfOpName:`TensorListScatter`,category:`control`,inputs:[{start:0,name:`tensor`,type:`tensor`},{start:1,name:`indices`,type:`number[]`},{start:2,name:`elementShape`,type:`shape`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListScatterV2`,category:`control`,inputs:[{start:0,name:`tensor`,type:`tensor`},{start:1,name:`indices`,type:`number[]`},{start:2,name:`elementShape`,type:`shape`},{start:3,name:`numElements`,type:`number`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListGather`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`},{start:1,name:`indices`,type:`number[]`},{start:2,name:`elementShape`,type:`shape`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListGetItem`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`},{start:1,name:`index`,type:`number`},{start:2,name:`elementShape`,type:`shape`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListSetItem`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`},{start:1,name:`index`,type:`number`},{start:2,name:`tensor`,type:`tensor`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListReserve`,category:`control`,inputs:[{start:0,name:`elementShape`,type:`shape`},{start:1,name:`numElements`,type:`number`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListFromTensor`,category:`control`,inputs:[{start:0,name:`tensor`,type:`tensor`},{start:1,name:`elementShape`,type:`shape`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListStack`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`},{start:1,name:`elementShape`,type:`shape`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`},{tfName:`num_elements`,name:`numElements`,type:`dtype`}]},{tfOpName:`TensorListSplit`,category:`control`,inputs:[{start:0,name:`tensor`,type:`tensor`},{start:1,name:`elementShape`,type:`shape`},{start:2,name:`lengths`,type:`number[]`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListConcat`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`}],attrs:[{tfName:`element_shape`,name:`elementShape`,type:`shape`},{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListConcatV2`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`}],attrs:[{tfName:`element_shape`,name:`elementShape`,type:`shape`},{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListPopBack`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`},{start:1,name:`elementShape`,type:`shape`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListPushBack`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`},{start:1,name:`tensor`,type:`tensor`}],attrs:[{tfName:`element_dtype`,name:`elementDType`,type:`dtype`}]},{tfOpName:`TensorListLength`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`}]},{tfOpName:`TensorListResize`,category:`control`,inputs:[{start:0,name:`tensorListId`,type:`tensor`},{start:1,name:`size`,type:`number`}]}],WL=t({json:()=>GL}),GL=[{tfOpName:`AvgPool`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0},{tfName:`ksize`,name:`kernelSize`,type:`number[]`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`MaxPool`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0},{tfName:`ksize`,name:`kernelSize`,type:`number[]`},{tfName:`explicit_paddings`,name:`explicitPaddings`,type:`number[]`,defaultValue:[],notSupported:!0},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`MaxPoolWithArgmax`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`ksize`,name:`kernelSize`,type:`number[]`},{tfName:`include_batch_in_index`,name:`includeBatchInIndex`,type:`bool`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`AvgPool3D`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0},{tfName:`ksize`,name:`kernelSize`,type:`number[]`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`MaxPool3D`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0},{tfName:`ksize`,name:`kernelSize`,type:`number[]`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Conv1D`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`filter`,type:`tensor`}],attrs:[{tfName:`stride`,name:`stride`,type:`number`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,defaultValue:`NWC`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`dilation`,name:`dilation`,type:`number`,defaultValue:1}]},{tfOpName:`Conv2D`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`filter`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`useCudnnOnGpu`,name:`useCudnnOnGpu`,type:`bool`},{tfName:`data_format`,name:`dataFormat`,type:`string`,defaultValue:`NHWC`},{tfName:`explicit_paddings`,name:`explicitPaddings`,type:`number[]`,defaultValue:[]},{tfName:`dilations`,name:`dilations`,type:`number[]`}]},{tfOpName:`_FusedConv2D`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`filter`,type:`tensor`},{start:2,end:0,name:`args`,type:`tensors`}],attrs:[{tfName:`num_args`,name:`numArgs`,type:`number`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`explicit_paddings`,name:`explicitPaddings`,type:`number[]`,defaultValue:[]},{tfName:`use_cudnn_on_gpu`,name:`useCudnnOnGpu`,type:`bool`,defaultValue:!0},{tfName:`data_format`,name:`dataFormat`,type:`string`,defaultValue:`NHWC`},{tfName:`dilations`,name:`dilations`,type:`number[]`,defaultValue:[1,1,1,1]},{tfName:`fused_ops`,name:`fusedOps`,type:`string[]`,defaultValue:[]},{tfName:`epsilon`,name:`epsilon`,type:`number`,defaultValue:1e-4},{tfName:`leakyrelu_alpha`,name:`leakyreluAlpha`,type:`number`,defaultValue:.2}]},{tfOpName:`Conv2DBackpropInput`,category:`convolution`,inputs:[{start:2,name:`x`,type:`tensor`},{start:1,name:`filter`,type:`tensor`},{start:0,name:`outputShape`,type:`number[]`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0},{tfName:`explicit_paddings`,name:`explicitPaddings`,type:`number[]`,defaultValue:[]},{tfName:`dilations`,name:`dilations`,type:`number[]`,notSupported:!0}]},{tfOpName:`DepthwiseConv2d`,category:`convolution`,inputs:[{start:0,name:`input`,type:`tensor`},{start:1,name:`filter`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,defaultValue:`NHWC`},{tfName:`explicit_paddings`,name:`explicitPaddings`,type:`number[]`,defaultValue:[]},{tfName:`dilations`,name:`dilations`,type:`number[]`}]},{tfOpName:`DepthwiseConv2dNative`,category:`convolution`,inputs:[{start:0,name:`input`,type:`tensor`},{start:1,name:`filter`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,defaultValue:`NHWC`},{tfName:`explicit_paddings`,name:`explicitPaddings`,type:`number[]`,defaultValue:[]},{tfName:`dilations`,name:`dilations`,type:`number[]`}]},{tfOpName:`FusedDepthwiseConv2dNative`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`filter`,type:`tensor`},{start:2,end:0,name:`args`,type:`tensors`}],attrs:[{tfName:`num_args`,name:`numArgs`,type:`number`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,defaultValue:`NHWC`},{tfName:`dilations`,name:`dilations`,type:`number[]`,defaultValue:[1,1,1,1]},{tfName:`fused_ops`,name:`fusedOps`,type:`string[]`,defaultValue:[]},{tfName:`explicit_paddings`,name:`explicitPaddings`,type:`number[]`,defaultValue:[]}]},{tfOpName:`Conv3D`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`filter`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`},{tfName:`data_format`,name:`dataFormat`,type:`string`,defaultValue:`NHWC`},{tfName:`dilations`,name:`dilations`,type:`number[]`}]},{tfOpName:`Dilation2D`,category:`convolution`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`filter`,type:`tensor`}],attrs:[{tfName:`strides`,name:`strides`,type:`number[]`},{tfName:`rates`,name:`dilations`,type:`number[]`},{tfName:`padding`,name:`pad`,type:`string`}]}],KL=t({json:()=>qL}),qL=[{tfOpName:`Fill`,category:`creation`,inputs:[{start:0,name:`shape`,type:`number[]`},{start:1,name:`value`,type:`number`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`LinSpace`,category:`creation`,inputs:[{start:0,name:`start`,type:`number`},{start:1,name:`stop`,type:`number`},{start:2,name:`num`,type:`number`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`OneHot`,category:`creation`,inputs:[{start:0,name:`indices`,type:`tensor`},{start:1,name:`depth`,type:`number`},{start:2,name:`onValue`,type:`number`,defaultValue:1},{start:3,name:`offValue`,type:`number`,defaultValue:0}],attrs:[{tfName:`axis`,name:`axis`,type:`number`,notSupported:!0},{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`Ones`,category:`creation`,inputs:[{start:0,name:`shape`,type:`number[]`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`OnesLike`,category:`creation`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`dtype`,name:`dtype`,type:`dtype`}]},{tfOpName:`RandomStandardNormal`,category:`creation`,inputs:[{start:0,name:`shape`,type:`number[]`}],attrs:[{tfName:`seed`,name:`seed`,type:`number`,defaultValue:0},{tfName:`seed2`,name:`seed2`,type:`number`,defaultValue:0,notSupported:!0},{tfName:`dtype`,name:`dtype`,type:`dtype`},{tfName:`T`,name:`T`,type:`number`,notSupported:!0}]},{tfOpName:`RandomUniform`,category:`creation`,inputs:[{start:0,name:`shape`,type:`number[]`}],attrs:[{tfName:`minval`,name:`minval`,type:`number`,defaultValue:0},{tfName:`maxval`,name:`maxval`,type:`number`,defaultValue:1},{tfName:`dtype`,name:`dtype`,type:`dtype`},{tfName:`seed`,name:`seed`,type:`number`,defaultValue:0},{tfName:`seed2`,name:`seed2`,type:`number`,defaultValue:0,notSupported:!0},{tfName:`T`,name:`T`,type:`number`,notSupported:!0}]},{tfOpName:`Range`,category:`creation`,inputs:[{start:0,name:`start`,type:`number`},{start:1,name:`stop`,type:`number`},{start:2,name:`step`,type:`number`,defaultValue:0}],attrs:[{tfName:`Tidx`,name:`dtype`,type:`dtype`}]},{tfOpName:`TruncatedNormal`,category:`creation`,inputs:[{start:0,name:`shape`,type:`number[]`}],attrs:[{tfName:`means`,name:`mean`,type:`number`,defaultValue:0},{tfName:`stddev`,name:`stdDev`,type:`number`,defaultValue:1},{tfName:`seed`,name:`seed`,type:`number`},{tfName:`seed2`,name:`seed2`,type:`number`,defaultValue:0,notSupported:!0},{tfName:`dtype`,name:`dtype`,type:`dtype`},{tfName:`T`,name:`T`,type:`number`,notSupported:!0}]},{tfOpName:`Zeros`,category:`creation`,inputs:[{start:0,name:`shape`,type:`number[]`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`ZerosLike`,category:`creation`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`}]},{tfOpName:`Multinomial`,category:`creation`,inputs:[{start:0,name:`logits`,type:`tensor`},{start:1,name:`numSamples`,type:`number`}],attrs:[{tfName:`seed`,name:`seed`,type:`number`},{tfName:`seed2`,name:`seed2`,type:`number`},{tfName:`T`,name:`dtype`,type:`dtype`},{tfName:`output_dtype`,name:`output_dtype`,type:`dtype`}]}],JL=t({json:()=>YL}),YL=[{tfOpName:`NonMaxSuppressionV2`,category:`dynamic`,inputs:[{start:0,name:`boxes`,type:`tensor`},{start:1,name:`scores`,type:`tensor`},{start:2,name:`maxOutputSize`,type:`number`},{start:3,name:`iouThreshold`,type:`number`}]},{tfOpName:`NonMaxSuppressionV3`,category:`dynamic`,inputs:[{start:0,name:`boxes`,type:`tensor`},{start:1,name:`scores`,type:`tensor`},{start:2,name:`maxOutputSize`,type:`number`},{start:3,name:`iouThreshold`,type:`number`},{start:4,name:`scoreThreshold`,type:`number`}]},{tfOpName:`NonMaxSuppressionV4`,category:`dynamic`,inputs:[{start:0,name:`boxes`,type:`tensor`},{start:1,name:`scores`,type:`tensor`},{start:2,name:`maxOutputSize`,type:`number`},{start:3,name:`iouThreshold`,type:`number`},{start:4,name:`scoreThreshold`,type:`number`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0},{tfName:`T_threshold`,name:`threshold`,type:`dtype`,notSupported:!0},{tfName:`pad_to_max_output_size`,name:`padToMaxOutputSize`,type:`bool`}]},{tfOpName:`NonMaxSuppressionV5`,category:`dynamic`,inputs:[{start:0,name:`boxes`,type:`tensor`},{start:1,name:`scores`,type:`tensor`},{start:2,name:`maxOutputSize`,type:`number`},{start:3,name:`iouThreshold`,type:`number`},{start:4,name:`scoreThreshold`,type:`number`},{start:5,name:`softNmsSigma`,type:`number`}]},{tfOpName:`Where`,category:`dynamic`,inputs:[{start:0,name:`condition`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`ListDiff`,category:`dynamic`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`y`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]}],XL=t({json:()=>ZL}),ZL=[{tfOpName:`LowerBound`,category:`evaluation`,inputs:[{start:0,name:`sortedSequence`,type:`tensor`},{start:1,name:`values`,type:`tensor`}]},{tfOpName:`TopKV2`,category:`evaluation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`k`,type:`number`}],attrs:[{tfName:`sorted`,name:`sorted`,type:`bool`}]},{tfOpName:`UpperBound`,category:`evaluation`,inputs:[{start:0,name:`sortedSequence`,type:`tensor`},{start:1,name:`values`,type:`tensor`}]},{tfOpName:`Unique`,category:`evaluation`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`UniqueV2`,category:`evaluation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number`}]}],QL=t({json:()=>$L}),$L=[{tfOpName:`PlaceholderWithDefault`,category:`graph`,inputs:[{start:0,name:`default`,type:`tensor`}],attrs:[{tfName:`shape`,name:`shape`,type:`shape`},{tfName:`dtype`,name:`dtype`,type:`dtype`}]},{tfOpName:`Placeholder`,category:`graph`,attrs:[{tfName:`shape`,name:`shape`,type:`shape`},{tfName:`dtype`,name:`dtype`,type:`dtype`}]},{tfOpName:`Const`,category:`graph`},{tfOpName:`Identity`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`IdentityN`,category:`graph`,inputs:[{start:0,end:0,name:`x`,type:`tensors`}]},{tfOpName:`Snapshot`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`Rank`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`Size`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`Shape`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`ShapeN`,category:`graph`,inputs:[{start:0,end:0,name:`x`,type:`tensors`}]},{tfOpName:`Print`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`data`,type:`tensors`}],attrs:[{tfName:`message`,name:`message`,type:`string`},{tfName:`first_n`,name:`firstN`,type:`number`,notSupported:!0},{tfName:`summarize`,name:`summarize`,type:`number`,defaultValue:3}]},{tfOpName:`NoOp`,category:`graph`,inputs:[]},{tfOpName:`StopGradient`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`FakeQuantWithMinMaxVars`,category:`graph`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`min`,name:`min`,type:`number`},{tfName:`max`,name:`max`,type:`number`}]}],eR=t({json:()=>tR}),tR=[{tfOpName:`HashTable`,category:`hash_table`,inputs:[],attrs:[{tfName:`shared_name`,name:`sharedName`,type:`string`},{tfName:`use_node_name_sharing`,name:`useNodeNameSharing`,type:`bool`},{tfName:`key_dtype`,name:`keyDType`,type:`dtype`},{tfName:`value_dtype`,name:`valueDType`,type:`dtype`}]},{tfOpName:`HashTableV2`,category:`hash_table`,inputs:[],attrs:[{tfName:`shared_name`,name:`sharedName`,type:`string`},{tfName:`use_node_name_sharing`,name:`useNodeNameSharing`,type:`bool`},{tfName:`key_dtype`,name:`keyDType`,type:`dtype`},{tfName:`value_dtype`,name:`valueDType`,type:`dtype`}]},{tfOpName:`LookupTableImport`,category:`hash_table`,inputs:[{start:0,name:`tableHandle`,type:`tensor`},{start:1,name:`keys`,type:`tensor`},{start:2,name:`values`,type:`tensor`}],attrs:[{tfName:`Tin`,name:`tIn`,type:`dtype`,notSupported:!0},{tfName:`Tout`,name:`tOut`,type:`dtype`,notSupported:!0}]},{tfOpName:`LookupTableImportV2`,category:`hash_table`,inputs:[{start:0,name:`tableHandle`,type:`tensor`},{start:1,name:`keys`,type:`tensor`},{start:2,name:`values`,type:`tensor`}],attrs:[{tfName:`Tin`,name:`tIn`,type:`dtype`,notSupported:!0},{tfName:`Tout`,name:`tOut`,type:`dtype`,notSupported:!0}]},{tfOpName:`LookupTableFind`,category:`hash_table`,inputs:[{start:0,name:`tableHandle`,type:`tensor`},{start:1,name:`keys`,type:`tensor`},{start:2,name:`defaultValue`,type:`tensor`}],attrs:[{tfName:`Tin`,name:`tIn`,type:`dtype`,notSupported:!0},{tfName:`Tout`,name:`tOut`,type:`dtype`,notSupported:!0}]},{tfOpName:`LookupTableFindV2`,category:`hash_table`,inputs:[{start:0,name:`tableHandle`,type:`tensor`},{start:1,name:`keys`,type:`tensor`},{start:2,name:`defaultValue`,type:`tensor`}],attrs:[{tfName:`Tin`,name:`tIn`,type:`dtype`,notSupported:!0},{tfName:`Tout`,name:`tOut`,type:`dtype`,notSupported:!0}]},{tfOpName:`LookupTableSize`,category:`hash_table`,inputs:[{start:0,name:`tableHandle`,type:`tensor`}]},{tfOpName:`LookupTableSizeV2`,category:`hash_table`,inputs:[{start:0,name:`tableHandle`,type:`tensor`}]}],nR=t({json:()=>rR}),rR=[{tfOpName:`ResizeBilinear`,category:`image`,inputs:[{start:0,name:`images`,type:`tensor`},{start:1,name:`size`,type:`number[]`}],attrs:[{tfName:`align_corners`,name:`alignCorners`,type:`bool`},{tfName:`half_pixel_centers`,name:`halfPixelCenters`,type:`bool`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`ResizeNearestNeighbor`,category:`image`,inputs:[{start:0,name:`images`,type:`tensor`},{start:1,name:`size`,type:`number[]`}],attrs:[{tfName:`align_corners`,name:`alignCorners`,type:`bool`},{tfName:`half_pixel_centers`,name:`halfPixelCenters`,type:`bool`},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`CropAndResize`,category:`image`,inputs:[{start:0,name:`image`,type:`tensor`},{start:1,name:`boxes`,type:`tensor`},{start:2,name:`boxInd`,type:`tensor`},{start:3,name:`cropSize`,type:`number[]`}],attrs:[{tfName:`method`,name:`method`,type:`string`},{tfName:`extrapolation_value`,name:`extrapolationValue`,type:`number`}]},{tfOpName:`ImageProjectiveTransformV3`,category:`image`,inputs:[{start:0,name:`images`,type:`tensor`},{start:1,name:`transforms`,type:`tensor`},{start:2,name:`outputShape`,type:`number[]`},{start:3,name:`fillValue`,type:`number`}],attrs:[{tfName:`interpolation`,name:`interpolation`,type:`string`},{tfName:`fill_mode`,name:`fillMode`,type:`string`}]}],iR=t({json:()=>aR}),aR=[{tfOpName:`Equal`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`NotEqual`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Greater`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`GreaterEqual`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Less`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`LessEqual`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`LogicalAnd`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`LogicalNot`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`LogicalOr`,category:`logical`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Select`,category:`logical`,inputs:[{start:0,name:`condition`,type:`tensor`},{start:1,name:`a`,type:`tensor`},{start:2,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`SelectV2`,category:`logical`,inputs:[{start:0,name:`condition`,type:`tensor`},{start:1,name:`a`,type:`tensor`},{start:2,name:`b`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]}],oR=t({json:()=>sR}),sR=[{tfOpName:`_FusedMatMul`,category:`matrices`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`},{start:2,end:0,name:`args`,type:`tensors`}],attrs:[{tfName:`num_args`,name:`numArgs`,type:`number`},{tfName:`fused_ops`,name:`fusedOps`,type:`string[]`,defaultValue:[]},{tfName:`epsilon`,name:`epsilon`,type:`number`,defaultValue:1e-4},{tfName:`transpose_a`,name:`transposeA`,type:`bool`,defaultValue:!1},{tfName:`transpose_b`,name:`transposeB`,type:`bool`,defaultValue:!1},{tfName:`leakyrelu_alpha`,name:`leakyreluAlpha`,type:`number`,defaultValue:.2},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`MatMul`,category:`matrices`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`transpose_a`,name:`transposeA`,type:`bool`,defaultValue:!1},{tfName:`transpose_b`,name:`transposeB`,type:`bool`,defaultValue:!1},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`BatchMatMul`,category:`matrices`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`adj_x`,name:`transposeA`,type:`bool`,defaultValue:!1},{tfName:`adj_y`,name:`transposeB`,type:`bool`,defaultValue:!1},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`BatchMatMulV2`,category:`matrices`,inputs:[{start:0,name:`a`,type:`tensor`},{start:1,name:`b`,type:`tensor`}],attrs:[{tfName:`adj_x`,name:`transposeA`,type:`bool`,defaultValue:!1},{tfName:`adj_y`,name:`transposeB`,type:`bool`,defaultValue:!1},{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Transpose`,category:`matrices`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`perm`,type:`number[]`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`Einsum`,category:`matrices`,inputs:[{start:0,end:0,name:`tensors`,type:`tensors`}],attrs:[{tfName:`equation`,name:`equation`,type:`string`},{tfName:`N`,name:`n`,type:`number`,defaultValue:2},{tfName:`T`,name:`dtype`,type:`dtype`}]}],cR=t({json:()=>lR}),lR=[{tfOpName:`EuclideanNorm`,category:`normalization`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`,defaultValue:!1}]},{tfOpName:`FusedBatchNorm`,category:`normalization`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`scale`,type:`tensor`},{start:2,name:`offset`,type:`tensor`},{start:3,name:`mean`,type:`tensor`},{start:4,name:`variance`,type:`tensor`}],attrs:[{tfName:`epsilon`,name:`epsilon`,type:`number`,defaultValue:.001},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0}]},{tfOpName:`FusedBatchNormV2`,category:`normalization`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`scale`,type:`tensor`},{start:2,name:`offset`,type:`tensor`},{start:3,name:`mean`,type:`tensor`},{start:4,name:`variance`,type:`tensor`}],attrs:[{tfName:`epsilon`,name:`epsilon`,type:`number`,defaultValue:.001},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0}]},{tfOpName:`FusedBatchNormV3`,category:`normalization`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`scale`,type:`tensor`},{start:2,name:`offset`,type:`tensor`},{start:3,name:`mean`,type:`tensor`},{start:4,name:`variance`,type:`tensor`}],attrs:[{tfName:`epsilon`,name:`epsilon`,type:`number`,defaultValue:.001},{tfName:`data_format`,name:`dataFormat`,type:`string`,notSupported:!0}]},{tfOpName:`LRN`,category:`normalization`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`depth_radius`,name:`radius`,type:`number`,defaultValue:5},{tfName:`bias`,name:`bias`,type:`number`,defaultValue:1},{tfName:`alpha`,name:`alpha`,type:`number`,defaultValue:1},{tfName:`beta`,name:`beta`,type:`number`,defaultValue:.5}]},{tfOpName:`Softmax`,category:`normalization`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`LogSoftmax`,category:`normalization`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`SparseToDense`,category:`normalization`,inputs:[{start:0,name:`sparseIndices`,type:`tensor`},{start:1,name:`outputShape`,type:`number[]`},{start:2,name:`sparseValues`,type:`tensor`},{start:3,name:`defaultValue`,type:`tensor`}],attrs:[{tfName:`validate_indices`,name:`validateIndices`,type:`bool`,defaultValue:!0,notSupported:!0}]}],uR=t({json:()=>dR}),dR=[{tfOpName:`Bincount`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`size`,type:`number`},{start:2,name:`weights`,type:`tensor`}]},{tfOpName:`DenseBincount`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`size`,type:`number`},{start:2,name:`weights`,type:`tensor`}],attrs:[{tfName:`binary_output`,name:`binaryOutput`,type:`bool`}]},{tfOpName:`Max`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`}]},{tfOpName:`Mean`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`}]},{tfOpName:`Min`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`}]},{tfOpName:`Sum`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`}]},{tfOpName:`All`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`}]},{tfOpName:`Any`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`}]},{tfOpName:`ArgMax`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number`}]},{tfOpName:`ArgMin`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number`}]},{tfOpName:`Prod`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}],attrs:[{tfName:`keep_dims`,name:`keepDims`,type:`bool`}]},{tfOpName:`Cumprod`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number`}],attrs:[{tfName:`exclusive`,name:`exclusive`,type:`bool`},{tfName:`reverse`,name:`reverse`,type:`bool`}]},{tfOpName:`Cumsum`,category:`reduction`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number`}],attrs:[{tfName:`exclusive`,name:`exclusive`,type:`bool`},{tfName:`reverse`,name:`reverse`,type:`bool`}]}],fR=t({json:()=>pR}),pR=[{tfOpName:`ConcatV2`,category:`slice_join`,inputs:[{start:0,end:-1,name:`tensors`,type:`tensors`},{start:-1,name:`axis`,type:`number`}],attrs:[{tfName:`N`,name:`n`,type:`number`,defaultValue:2}]},{tfOpName:`Concat`,category:`slice_join`,inputs:[{start:1,end:0,name:`tensors`,type:`tensors`},{start:0,name:`axis`,type:`number`}],attrs:[{tfName:`N`,name:`n`,type:`number`,defaultValue:2}]},{tfOpName:`GatherV2`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`indices`,type:`tensor`},{start:2,name:`axis`,type:`number`,defaultValue:0}],attrs:[{tfName:`batch_dims`,name:`batchDims`,type:`number`,defaultValue:0}]},{tfOpName:`Gather`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`indices`,type:`tensor`}],attrs:[{tfName:`validate_indices`,name:`validateIndices`,type:`bool`,notSupported:!0}]},{tfOpName:`Reverse`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`dims`,type:`bool[]`}]},{tfOpName:`ReverseV2`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number[]`}]},{tfOpName:`Slice`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`begin`,type:`number[]`},{start:2,name:`size`,type:`number[]`}]},{tfOpName:`StridedSlice`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`begin`,type:`number[]`},{start:2,name:`end`,type:`number[]`},{start:3,name:`strides`,type:`number[]`}],attrs:[{tfName:`begin_mask`,name:`beginMask`,type:`number`,defaultValue:0},{tfName:`end_mask`,name:`endMask`,type:`number`,defaultValue:0},{tfName:`new_axis_mask`,name:`newAxisMask`,type:`number`,defaultValue:0},{tfName:`ellipsis_mask`,name:`ellipsisMask`,type:`number`,defaultValue:0},{tfName:`shrink_axis_mask`,name:`shrinkAxisMask`,type:`number`,defaultValue:0}]},{tfOpName:`Pack`,category:`slice_join`,inputs:[{start:0,end:0,name:`tensors`,type:`tensors`}],attrs:[{tfName:`axis`,name:`axis`,type:`number`,defaultValue:0}]},{tfOpName:`Unpack`,category:`slice_join`,inputs:[{start:0,name:`tensor`,type:`tensor`}],attrs:[{tfName:`axis`,name:`axis`,type:`number`,defaultValue:0},{tfName:`num`,name:`num`,type:`number`,defaultValue:0,notSupported:!0}]},{tfOpName:`Tile`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`reps`,type:`number[]`}]},{tfOpName:`Split`,category:`slice_join`,inputs:[{start:0,name:`axis`,type:`number`,defaultValue:0},{start:1,name:`x`,type:`tensor`}],attrs:[{tfName:`num_split`,name:`numOrSizeSplits`,type:`number`,defaultValue:1}]},{tfOpName:`SplitV`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`numOrSizeSplits`,type:`number[]`},{start:2,name:`axis`,type:`number`,defaultValue:0}]},{tfOpName:`ScatterNd`,category:`slice_join`,inputs:[{start:0,name:`indices`,type:`tensor`},{start:1,name:`values`,type:`tensor`},{start:2,name:`shape`,type:`number[]`}]},{tfOpName:`GatherNd`,category:`slice_join`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`indices`,type:`tensor`}]},{tfOpName:`SparseToDense`,category:`slice_join`,inputs:[{start:0,name:`sparseIndices`,type:`tensor`},{start:1,name:`outputShape`,type:`number[]`},{start:2,name:`sparseValues`,type:`tensor`},{start:3,name:`defaultValue`,type:`tensor`}],attrs:[{tfName:`validate_indices`,name:`validateIndices`,type:`bool`,defaultValue:!1,notSupported:!0}]}],mR=t({json:()=>hR}),hR=[{tfOpName:`SparseFillEmptyRows`,category:`sparse`,inputs:[{start:0,name:`indices`,type:`tensor`},{start:1,name:`values`,type:`tensor`},{start:2,name:`denseShape`,type:`tensor`},{start:3,name:`defaultValue`,type:`tensor`}]},{tfOpName:`SparseReshape`,category:`sparse`,inputs:[{start:0,name:`inputIndices`,type:`tensor`},{start:1,name:`inputShape`,type:`tensor`},{start:2,name:`newShape`,type:`tensor`}],attrs:[{tfName:`T`,name:`dtype`,type:`dtype`,notSupported:!0}]},{tfOpName:`SparseSegmentMean`,category:`sparse`,inputs:[{start:0,name:`data`,type:`tensor`},{start:1,name:`indices`,type:`tensor`},{start:2,name:`segmentIds`,type:`tensor`}]},{tfOpName:`SparseSegmentSum`,category:`sparse`,inputs:[{start:0,name:`data`,type:`tensor`},{start:1,name:`indices`,type:`tensor`},{start:2,name:`segmentIds`,type:`tensor`}]}],gR=t({json:()=>_R}),_R=[{tfOpName:`FFT`,category:`spectral`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`IFFT`,category:`spectral`,inputs:[{start:0,name:`x`,type:`tensor`}]},{tfOpName:`RFFT`,category:`spectral`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`fft_length`,type:`number`,notSupported:!0}]},{tfOpName:`IRFFT`,category:`spectral`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`fft_length`,type:`number`,notSupported:!0}]}],vR=t({json:()=>yR}),yR=[{tfOpName:`StringNGrams`,category:`string`,inputs:[{start:0,name:`data`,type:`tensor`},{start:1,name:`dataSplits`,type:`tensor`}],attrs:[{tfName:`separator`,name:`separator`,type:`string`},{tfName:`ngram_widths`,name:`nGramWidths`,type:`number[]`},{tfName:`left_pad`,name:`leftPad`,type:`string`},{tfName:`right_pad`,name:`rightPad`,type:`string`},{tfName:`pad_width`,name:`padWidth`,type:`number`},{tfName:`preserve_short_sequences`,name:`preserveShortSequences`,type:`bool`}],outputs:[`ngrams`,`ngrams_splits`]},{tfOpName:`StringSplit`,category:`string`,inputs:[{start:0,name:`input`,type:`tensor`},{start:1,name:`delimiter`,type:`tensor`}],attrs:[{tfName:`skip_empty`,name:`skipEmpty`,type:`bool`}],outputs:[`indices`,`values`,`shape`]},{tfOpName:`StringToHashBucketFast`,category:`string`,inputs:[{start:0,name:`input`,type:`tensor`}],attrs:[{tfName:`num_buckets`,name:`numBuckets`,type:`number`}]}],bR=t({json:()=>xR}),xR=[{tfOpName:`Cast`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`SrcT`,name:`sdtype`,type:`dtype`,notSupported:!0},{tfName:`DstT`,name:`dtype`,type:`dtype`}]},{tfOpName:`ExpandDims`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`axis`,type:`number`}]},{tfOpName:`MirrorPad`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`padding`,type:`number[]`}],attrs:[{tfName:`mode`,name:`mode`,type:`string`}]},{tfOpName:`Pad`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`padding`,type:`number[]`}],attrs:[{tfName:`constant_value`,name:`constantValue`,type:`number`,defaultValue:0}]},{tfOpName:`PadV2`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`padding`,type:`number[]`},{start:2,name:`constantValue`,type:`number`,defaultValue:0}]},{tfOpName:`Reshape`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`shape`,type:`number[]`}]},{tfOpName:`Squeeze`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`axis`,tfDeprecatedName:`squeeze_dims`,name:`axis`,type:`number[]`}]},{tfOpName:`SpaceToBatchND`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`blockShape`,type:`number[]`},{start:2,name:`paddings`,type:`number[]`}]},{tfOpName:`BatchToSpaceND`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`blockShape`,type:`number[]`},{start:2,name:`crops`,type:`number[]`}]},{tfOpName:`DepthToSpace`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`}],attrs:[{tfName:`block_size`,name:`blockSize`,type:`number`},{tfName:`data_format`,name:`dataFormat`,type:`string`}]},{tfOpName:`BroadcastTo`,category:`transformation`,inputs:[{start:0,name:`x`,type:`tensor`},{start:1,name:`shape`,type:`number[]`}],attrs:[]},{tfOpName:`BroadcastArgs`,category:`transformation`,inputs:[{start:0,name:`s0`,type:`tensor`},{start:1,name:`s1`,type:`tensor`}],attrs:[]}],SR=class{static get Instance(){return this._instance||=new this}constructor(){let e=[RL,BL,HL,WL,KL,JL,XL,QL,eR,nR,iR,oR,cR,uR,fR,mR,gR,vR,bR],t=[].concat(...e.map(e=>e.json));this.opMappers=t.reduce((e,t)=>(e[t.tfOpName]=t,e),{})}transformGraph(e,t={}){let n=e.node,r=[],i=[],a=[],o=n.reduce((e,t)=>(e[t.name]=this.mapNode(t),t.op.startsWith(`Placeholder`)?r.push(e[t.name]):t.op===`Const`?i.push(e[t.name]):(t.input==null||t.input.length===0)&&a.push(e[t.name]),e),{}),s=[],c=[],l={},u={};t!=null&&(l=this.mapSignatureEntries(t.inputs),u=this.mapSignatureEntries(t.outputs));let d=Object.keys(o);d.forEach(e=>{let t=o[e];t.inputNames.forEach((e,n)=>{let[r,,i]=NL(e),a=o[r];if(a.outputs!=null){let e=a.outputs.indexOf(i);if(e!==-1){let i=`${r}:${e}`;t.inputNames[n]=i}}t.inputs.push(a),a.children.push(t)})}),Object.keys(u).length===0?d.forEach(e=>{let t=o[e];t.children.length===0&&c.push(t)}):Object.keys(u).forEach(e=>{let[t]=NL(e),n=o[t];n!=null&&(n.signatureKey=u[e],c.push(n))}),Object.keys(l).length>0?Object.keys(l).forEach(e=>{let[t]=NL(e),n=o[t];n&&(n.signatureKey=l[e],s.push(n))}):s=r;let f={};e.library!=null&&e.library.function!=null&&(f=e.library.function.reduce((e,t)=>(e[t.signature.name]=this.mapFunction(t),e),{}));let p={nodes:o,inputs:s,outputs:c,weights:i,placeholders:r,signature:t,functions:f};return a.length>0&&(p.initNodes=a),p}mapSignatureEntries(e){return Object.keys(e||{}).reduce((t,n)=>(t[e[n].name]=n,t),{})}mapNode(e){let t=AL(e.op)||this.opMappers[e.op]||{};e.attr??={};let n={name:e.name,op:e.op,category:t.category,inputNames:(e.input||[]).map(e=>e.startsWith(`^`)?e.slice(1):e),inputs:[],children:[],inputParams:{},attrParams:{},rawAttrs:e.attr,outputs:t.outputs};return t.inputs!=null&&(n.inputParams=t.inputs.reduce((e,t)=>(e[t.name]={type:t.type,inputIndexStart:t.start,inputIndexEnd:t.end},e),{})),t.attrs!=null&&(n.attrParams=t.attrs.reduce((t,n)=>{let r=n.type,i;switch(n.type){case`string`:i=TR(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=TR(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`string[]`:i=FR(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=FR(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`number`:i=DR(e.attr,n.tfName,n.defaultValue||0),i===void 0&&n.tfDeprecatedName&&(i=DR(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`number[]`:i=PR(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=PR(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`bool`:i=ER(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=ER(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`bool[]`:i=LR(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=LR(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`shape`:i=NR(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=NR(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`shape[]`:i=IR(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=IR(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`dtype`:i=AR(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=AR(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`dtype[]`:i=jR(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=jR(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`func`:i=kR(e.attr,n.tfName,n.defaultValue),i===void 0&&n.tfDeprecatedName&&(i=kR(e.attr,n.tfDeprecatedName,n.defaultValue));break;case`tensor`:case`tensors`:break;default:throw Error(`Unsupported param type: ${n.type} for op: ${e.op}`)}return t[n.name]={value:i,type:r},t},{})),n}mapFunction(e){let t=e.nodeDef,n=[],r=[],i={};t!=null&&(i=t.reduce((e,t)=>(e[t.name]=this.mapNode(t),t.op===`Const`&&r.push(e[t.name]),e),{}));let a=[],o=[];e.signature.inputArg.forEach(e=>{let[t]=NL(e.name),n={name:t,op:`Placeholder`,inputs:[],inputNames:[],category:`graph`,inputParams:{},attrParams:{dtype:{value:OR(e.type),type:`dtype`}},children:[]};n.signatureKey=e.name,a.push(n),i[t]=n}),Object.keys(i).forEach(e=>{let t=i[e];t.inputNames.forEach((e,n)=>{let[r,,a]=NL(e),o=i[r];if(o.outputs!=null){let e=o.outputs.indexOf(a);if(e!==-1){let i=`${r}:${e}`;t.inputNames[n]=i}}t.inputs.push(o),o.children.push(t)})});let s=e.ret;e.signature.outputArg.forEach(e=>{let[t,n]=NL(s[e.name]),r=i[t];r!=null&&(r.defaultOutput=n,o.push(r))});let c=this.mapArgsToSignature(e);return{nodes:i,inputs:a,outputs:o,weights:r,placeholders:n,signature:c}}mapArgsToSignature(e){return{methodName:e.signature.name,inputs:e.signature.inputArg.reduce((e,t)=>(e[t.name]=this.mapArgToTensorInfo(t),e),{}),outputs:e.signature.outputArg.reduce((t,n)=>(t[n.name]=this.mapArgToTensorInfo(n,e.ret),t),{})}}mapArgToTensorInfo(e,t){let n=e.name;return t!=null&&(n=t[n]),{name:n,dtype:e.type}}};function CR(e){let t=N().global;if(t.atob!==void 0)return t.atob(e);if(typeof Buffer<`u`)return new Buffer(e,`base64`).toString();throw Error(`Unable to decode base64 in this environment. Missing built-in atob() or Buffer()`)}function wR(e,t){let n=Array.isArray(e)?String.fromCharCode.apply(null,e):CR(e);return t?n:n.toLowerCase()}function TR(e,t,n,r=!1){let i=e[t];return i==null?n:wR(i.s,r)}function ER(e,t,n){let r=e[t];return r?r.b:n}function DR(e,t,n){let r=e[t]||{},i=r.i==null?r.f==null?n:r.f:r.i;return typeof i==`number`?i:parseInt(i,10)}function OR(e){switch(typeof e==`string`&&(e=DL[e]),e){case DL.DT_FLOAT:case DL.DT_HALF:return`float32`;case DL.DT_INT32:case DL.DT_INT64:case DL.DT_INT8:case DL.DT_UINT8:return`int32`;case DL.DT_BOOL:return`bool`;case DL.DT_DOUBLE:return`float32`;case DL.DT_STRING:return`string`;default:return null}}function kR(e,t,n){let r=e[t];return r&&r.func?r.func.name:n}function AR(e,t,n){let r=e[t];return r&&r.type?OR(r.type):n}function jR(e,t,n){let r=e[t];return r&&r.list&&r.list.type?r.list.type.map(e=>OR(e)):n}function MR(e){if(!e.unknownRank)return e.dim==null?[]:e.dim.map(e=>typeof e.size==`number`?e.size:parseInt(e.size,10))}function NR(e,t,n){let r=e[t];return r&&r.shape?MR(r.shape):n}function PR(e,t,n){let r=e[t];return r?((r.list.f&&r.list.f.length?r.list.f:r.list.i)||[]).map(e=>typeof e==`number`?e:parseInt(e,10)):n}function FR(e,t,n,r=!1){let i=e[t];return i&&i.list&&i.list.s?i.list.s.map(e=>wR(e,r)):n}function IR(e,t,n){let r=e[t];return r&&r.list&&r.list.shape?r.list.shape.map(e=>MR(e)):n}function LR(e,t,n){let r=e[t];return r&&r.list&&r.list.b?r.list.b:n}var RR=class{constructor(e,t,n){this.node=e,this.tensorMap=t,this.context=n,this.inputs=[],this.attrs={},this.inputs=e.inputNames.map(e=>this.getInput(e)),e.rawAttrs!=null&&(this.attrs=Object.keys(e.rawAttrs).reduce((e,t)=>(e[t]=this.getAttr(t),e),{}))}getInput(e){return jL(e,this.tensorMap,this.context)}getAttr(e,t){let n=this.node.rawAttrs[e];if(n.tensor!=null)return jL(e,this.tensorMap,this.context);if(n.i!=null||n.f!=null)return DR(this.node.rawAttrs,e,t);if(n.s!=null)return TR(this.node.rawAttrs,e,t);if(n.b!=null)return ER(this.node.rawAttrs,e,t);if(n.shape!=null)return NR(this.node.rawAttrs,e,t);if(n.type!=null)return AR(this.node.rawAttrs,e,t);if(n.list!=null){if(n.list.i!=null||n.list.f!=null)return PR(this.node.rawAttrs,e,t);if(n.list.s!=null)return FR(this.node.rawAttrs,e,t);if(n.list.shape!=null)return IR(this.node.rawAttrs,e,t);if(n.list.b!=null)return LR(this.node.rawAttrs,e,t);if(n.list.type!=null)return jR(this.node.rawAttrs,e,t)}return t}},zR=t({OP_SCOPE_SUFFIX:()=>Hi,abs:()=>zs,acos:()=>Vs,acosh:()=>Us,add:()=>Ms,addN:()=>Gs,all:()=>qs,any:()=>Ys,argMax:()=>Zs,argMin:()=>$s,asin:()=>tc,asinh:()=>rc,atan:()=>ac,atan2:()=>sc,atanh:()=>lc,avgPool:()=>Ac,avgPool3d:()=>Mc,basicLSTMCell:()=>Vc,batchNorm:()=>Kc,batchNorm2d:()=>Jc,batchNorm3d:()=>Xc,batchNorm4d:()=>Qc,batchToSpaceND:()=>Uc,bincount:()=>el,booleanMaskAsync:()=>Um,broadcastArgs:()=>nl,broadcastTo:()=>il,buffer:()=>L,cast:()=>io,ceil:()=>ol,clipByValue:()=>ll,clone:()=>oo,complex:()=>Wi,concat:()=>Pc,concat1d:()=>dl,concat2d:()=>pl,concat3d:()=>hl,concat4d:()=>_l,conv1d:()=>xl,conv2d:()=>yl,conv2dTranspose:()=>Tl,conv3d:()=>Dl,conv3dTranspose:()=>jl,cos:()=>Nl,cosh:()=>Fl,cosineWindow:()=>rh,cumprod:()=>Ll,cumsum:()=>zl,denseBincount:()=>Vl,depthToSpace:()=>Ul,depthwiseConv2d:()=>Gl,diag:()=>ql,dilation2d:()=>Yl,div:()=>Is,divNoNan:()=>ru,dot:()=>au,dropout:()=>th,einsum:()=>su,elu:()=>lu,enclosingPowerOfTwo:()=>nh,equal:()=>Zl,erf:()=>du,euclideanNorm:()=>Iu,exp:()=>Ru,expandDims:()=>Bu,expm1:()=>Hu,eye:()=>Ku,fft:()=>tm,fill:()=>sl,floor:()=>Ju,floorDiv:()=>Ps,fused:()=>Sh,gather:()=>Xu,gatherND:()=>Qm,greater:()=>Qu,greaterEqual:()=>ed,ifft:()=>rm,imag:()=>Xo,image:()=>c_,inTopKAsync:()=>ah,irfft:()=>am,isFinite:()=>nd,isInf:()=>id,isNaN:()=>od,leakyRelu:()=>cd,less:()=>ud,lessEqual:()=>fd,linalg:()=>l_,linspace:()=>pd,localResponseNormalization:()=>hd,log:()=>_d,log1p:()=>yd,logSigmoid:()=>wd,logSoftmax:()=>Dd,logSumExp:()=>kd,logicalAnd:()=>jd,logicalNot:()=>Nd,logicalOr:()=>Fd,logicalXor:()=>Ld,losses:()=>u_,lowerBound:()=>Vd,matMul:()=>R,max:()=>xu,maxPool:()=>Ud,maxPool3d:()=>Gd,maxPoolWithArgmax:()=>qd,maximum:()=>Yd,mean:()=>Zd,meshgrid:()=>ef,min:()=>Cu,minimum:()=>nf,mirrorPad:()=>af,mod:()=>sf,moments:()=>lf,movingAverage:()=>Gm,mul:()=>B,multiRNNCell:()=>df,multinomial:()=>pf,neg:()=>Qo,norm:()=>Pu,notEqual:()=>hf,oneHot:()=>zo,ones:()=>$d,onesLike:()=>_f,op:()=>I,outerProduct:()=>yf,pad:()=>xf,pad1d:()=>Cf,pad2d:()=>Tf,pad3d:()=>Df,pad4d:()=>kf,pool:()=>Ff,pow:()=>Tu,prelu:()=>Lf,print:()=>so,prod:()=>zf,raggedGather:()=>Vf,raggedTensorToTensor:()=>Uf,rand:()=>Gf,randomGamma:()=>ip,randomNormal:()=>op,randomStandardNormal:()=>cp,randomUniform:()=>up,range:()=>dp,real:()=>es,reciprocal:()=>pp,relu:()=>hp,relu6:()=>_p,reshape:()=>V,reverse:()=>yp,reverse1d:()=>xp,reverse2d:()=>Cp,reverse3d:()=>Tp,reverse4d:()=>Dp,rfft:()=>lm,round:()=>kp,rsqrt:()=>jp,scalar:()=>U,scatterND:()=>qm,searchSorted:()=>Bd,selu:()=>Np,separableConv2d:()=>Fp,setdiff1dAsync:()=>Lp,sigmoid:()=>Ic,sign:()=>zp,signal:()=>s_,sin:()=>Vp,sinh:()=>Up,slice:()=>H,slice1d:()=>Gp,slice2d:()=>qp,slice3d:()=>Yp,slice4d:()=>Zp,softmax:()=>$p,softplus:()=>Sd,spaceToBatchND:()=>jf,sparse:()=>d_,sparseToDense:()=>Xm,spectral:()=>o_,split:()=>sm,sqrt:()=>Du,square:()=>ku,squaredDifference:()=>dm,squeeze:()=>pm,stack:()=>hm,step:()=>_m,stridedSlice:()=>ym,string:()=>f_,sub:()=>W,sum:()=>ju,tan:()=>xm,tanh:()=>zc,tensor:()=>Ki,tensor1d:()=>Sm,tensor2d:()=>Cm,tensor3d:()=>as,tensor4d:()=>wm,tensor5d:()=>Tm,tensor6d:()=>Em,tile:()=>Wu,topk:()=>Om,transpose:()=>ns,truncatedNormal:()=>Am,unique:()=>Mm,unsortedSegmentSum:()=>Pm,unstack:()=>Im,upperBound:()=>Lm,variable:()=>Rm,where:()=>$l,whereAsync:()=>Vm,zeros:()=>Qd,zerosLike:()=>tu}),BR=(e,t,n,r=zR)=>{switch(e.op){case`BiasAdd`:case`AddV2`:case`Add`:return[r.add($(`a`,e,t,n),$(`b`,e,t,n))];case`AddN`:return[r.addN($(`tensors`,e,t,n))];case`FloorMod`:case`Mod`:return[r.mod($(`a`,e,t,n),$(`b`,e,t,n))];case`Mul`:return[r.mul($(`a`,e,t,n),$(`b`,e,t,n))];case`RealDiv`:case`Div`:return[r.div($(`a`,e,t,n),$(`b`,e,t,n))];case`DivNoNan`:return[r.divNoNan($(`a`,e,t,n),$(`b`,e,t,n))];case`FloorDiv`:return[r.floorDiv($(`a`,e,t,n),$(`b`,e,t,n))];case`Sub`:return[r.sub($(`a`,e,t,n),$(`b`,e,t,n))];case`Minimum`:return[r.minimum($(`a`,e,t,n),$(`b`,e,t,n))];case`Maximum`:return[r.maximum($(`a`,e,t,n),$(`b`,e,t,n))];case`Pow`:return[r.pow($(`a`,e,t,n),$(`b`,e,t,n))];case`SquaredDifference`:return[r.squaredDifference($(`a`,e,t,n),$(`b`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},VR=(e,t,n,r=zR)=>{switch(e.op){case`Abs`:case`ComplexAbs`:return[r.abs($(`x`,e,t,n))];case`Acos`:return[r.acos($(`x`,e,t,n))];case`Acosh`:return[r.acosh($(`x`,e,t,n))];case`Asin`:return[r.asin($(`x`,e,t,n))];case`Asinh`:return[r.asinh($(`x`,e,t,n))];case`Atan`:return[r.atan($(`x`,e,t,n))];case`Atan2`:return[r.atan2($(`x`,e,t,n),$(`y`,e,t,n))];case`Atanh`:return[r.atanh($(`x`,e,t,n))];case`Ceil`:return[r.ceil($(`x`,e,t,n))];case`Complex`:return[r.complex($(`real`,e,t,n),$(`imag`,e,t,n))];case`Cos`:return[r.cos($(`x`,e,t,n))];case`Cosh`:return[r.cosh($(`x`,e,t,n))];case`Elu`:return[r.elu($(`x`,e,t,n))];case`Erf`:return[r.erf($(`x`,e,t,n))];case`Exp`:return[r.exp($(`x`,e,t,n))];case`Expm1`:return[r.expm1($(`x`,e,t,n))];case`Floor`:return[r.floor($(`x`,e,t,n))];case`Log`:return[r.log($(`x`,e,t,n))];case`Log1p`:return[r.log1p($(`x`,e,t,n))];case`Imag`:return[r.imag($(`x`,e,t,n))];case`Neg`:return[r.neg($(`x`,e,t,n))];case`Reciprocal`:return[r.reciprocal($(`x`,e,t,n))];case`Real`:return[r.real($(`x`,e,t,n))];case`Relu`:return[r.relu($(`x`,e,t,n))];case`Round`:return[r.round($(`x`,e,t,n))];case`Selu`:return[r.selu($(`x`,e,t,n))];case`Sigmoid`:return[r.sigmoid($(`x`,e,t,n))];case`Sin`:return[r.sin($(`x`,e,t,n))];case`Sign`:return[r.sign($(`x`,e,t,n))];case`Sinh`:return[r.sinh($(`x`,e,t,n))];case`Softplus`:return[r.softplus($(`x`,e,t,n))];case`Sqrt`:return[r.sqrt($(`x`,e,t,n))];case`Square`:return[r.square($(`x`,e,t,n))];case`Tanh`:return[r.tanh($(`x`,e,t,n))];case`Tan`:return[r.tan($(`x`,e,t,n))];case`ClipByValue`:return[r.clipByValue($(`x`,e,t,n),$(`clipValueMin`,e,t,n),$(`clipValueMax`,e,t,n))];case`Relu6`:return[r.relu6($(`x`,e,t,n))];case`Rsqrt`:return[r.rsqrt(jL(e.inputNames[0],t,n))];case`Prod`:return[r.prod($(`x`,e,t,n),$(`axes`,e,t,n))];case`LeakyRelu`:return[r.leakyRelu($(`x`,e,t,n),$(`alpha`,e,t,n))];case`Prelu`:return[r.prelu($(`x`,e,t,n),$(`alpha`,e,t,n))];case`IsNan`:return[r.isNaN(jL(e.inputNames[0],t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};function HR(e,t,n=``){if(typeof e!=`number`&&typeof t!=`number`){f(e.length===t.length,()=>n+` Shapes ${e} and ${t} must match`);for(let r=0;r<e.length;r++){let i=e[r],a=t[r];f(i<0||a<0||i===a,()=>n+` Shapes ${e} and ${t} must match`)}}}function UR(e){return!(typeof e==`number`||e.some(e=>e<0))}function WR(e,t,n){let r=GR(e,n),i=!UR(r);if(i&&t.length===0)throw Error(`Tried to calculate elements of an empty list with non-fully-defined elementShape: ${r}`);if(i&&t.forEach(e=>{r=GR(e.shape,r)}),!UR(r))throw Error(`Non-fully-defined elementShape: ${r}`);return r}function GR(e,t){if(typeof e==`number`)return t;if(typeof t==`number`)return e;if(e.length!==t.length)throw Error(`Incompatible ranks during merge: ${e} vs. ${t}`);let n=[];for(let r=0;r<e.length;++r){let i=e[r],a=t[r];if(i>=0&&a>=0&&i!==a)throw Error(`Incompatible shape during merge: ${e} vs. ${t}`);n[r]=i>=0?i:a}return n}var KR=class{constructor(e,t,n,r,i,a,o){this.name=e,this.dtype=t,this.maxSize=n,this.elementShape=r,this.identicalElementShapes=i,this.dynamicSize=a,this.clearAfterRead=o,this.tensors=[],this.closed_=!1,this.idTensor=U(0),Uo(this.idTensor)}get id(){return this.idTensor.id}get closed(){return this.closed_}clearAndClose(e){this.tensors.forEach(t=>{e!=null&&e.has(t.tensor.id)||t.tensor.dispose()}),this.tensors=[],this.closed_=!0,this.idTensor.dispose()}size(){return this.tensors.length}read(e){if(this.closed_)throw Error(`TensorArray ${this.name} has already been closed.`);if(e<0||e>=this.size())throw Error(`Tried to read from index ${e}, but array size is: ${this.size()}`);let t=this.tensors[e];if(t.cleared)throw Error(`TensorArray ${this.name}: Could not read index ${e} twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?).`);return this.clearAfterRead&&(t.cleared=!0),t.read=!0,t.tensor}readMany(e){return e.map(e=>this.read(e))}write(e,t){if(this.closed_)throw Error(`TensorArray ${this.name} has already been closed.`);if(e<0||!this.dynamicSize&&e>=this.maxSize)throw Error(`Tried to write to index ${e}, but array is not resizeable and size is: ${this.maxSize}`);let n=this.tensors[e]||{};if(t.dtype!==this.dtype)throw Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e},
          because the value dtype is ${t.dtype}, but TensorArray dtype is ${this.dtype}.`);if(this.size()===0&&(this.elementShape==null||this.elementShape.length===0)&&(this.elementShape=t.shape),HR(this.elementShape,t.shape,`TensorArray ${this.name}: Could not write to TensorArray index ${e}.`),n.read)throw Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been read.`);if(n.written)throw Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been written.`);n.tensor=t,Uo(t),n.written=!0,this.tensors[e]=n}writeMany(e,t){if(e.length!==t.length)throw Error(`TensorArray ${this.name}: could not write multiple tensors,because the index size: ${e.length} is not the same as tensors size: ${t.length}.`);e.forEach((e,n)=>this.write(e,t[n]))}gather(e,t){if(t&&t!==this.dtype)throw Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${t}`);if(e)e=e.slice(0,this.size());else{e=[];for(let t=0;t<this.size();t++)e.push(t)}if(e.length===0)return Ki([],[0].concat(this.elementShape));let n=this.readMany(e);return HR(this.elementShape,n[0].shape,`TensorArray shape mismatch: `),hm(n,0)}concat(e){if(e&&e!==this.dtype)throw Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${e}`);if(this.size()===0)return Ki([],[0].concat(this.elementShape));let t=[];for(let e=0;e<this.size();e++)t.push(e);let n=this.readMany(t);return HR(this.elementShape,n[0].shape,`TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${n[0].shape})`),Pc(n,0)}scatter(e,t){if(t.dtype!==this.dtype)throw Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);if(e.length!==t.shape[0])throw Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${t.shape[0]}`);let n=Math.max(...e);if(!this.dynamicSize&&n>=this.maxSize)throw Error(`Max index must be < array size (${n}  vs. ${this.maxSize})`);this.writeMany(e,Im(t,0))}split(e,t){if(t.dtype!==this.dtype)throw Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);let n=0,r=e.map(e=>(n+=e,n));if(n!==t.shape[0])throw Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${n}, and tensor's shape is: ${t.shape}`);if(!this.dynamicSize&&e.length!==this.maxSize)throw Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${e.length}), and the TensorArray is not marked as dynamically resizeable`);let i=n===0?0:t.size/n,a=[];Vo(()=>{t=V(t,[1,n,i]);for(let n=0;n<e.length;++n){let o=[0,n===0?0:r[n-1],0],s=[1,e[n],i];a[n]=V(H(t,o,s),this.elementShape)}return a});let o=[];for(let t=0;t<e.length;t++)o[t]=t;this.writeMany(o,a)}},qR=class e{constructor(e,t,n,r=-1){this.tensors=e,this.elementShape=t,this.elementDtype=n,e?.forEach(e=>{if(n!==e.dtype)throw Error(`Invalid data types; op elements ${n}, but list elements ${e.dtype}`);HR(t,e.shape,`TensorList shape mismatch: `),Uo(e)}),this.idTensor=U(0),this.maxNumElements=r,Uo(this.idTensor)}get id(){return this.idTensor.id}copy(){return new e([...this.tensors],this.elementShape,this.elementDtype)}clearAndClose(e){this.tensors.forEach(t=>{e!=null&&e.has(t.id)||t.dispose()}),this.tensors.length=0,this.idTensor.dispose()}size(){return this.tensors.length}stack(e,t,n=-1){if(t!==this.elementDtype)throw Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(n!==-1&&this.tensors.length!==n)throw Error(`Operation expected a list with ${n} elements but got a list with ${this.tensors.length} elements.`);HR(e,this.elementShape,`TensorList shape mismatch: `);let r=WR(this.elementShape,this.tensors,e);return Vo(()=>hm(this.tensors.map(e=>V(e,r)),0))}popBack(e,t){if(t!==this.elementDtype)throw Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(this.size()===0)throw Error(`Trying to pop from an empty list.`);let n=WR(this.elementShape,this.tensors,e),r=this.tensors.pop();return r.kept=!1,HR(r.shape,e,`TensorList shape mismatch: `),V(r,n)}pushBack(e){if(e.dtype!==this.elementDtype)throw Error(`Invalid data types; op elements ${e.dtype}, but list elements ${this.elementDtype}`);if(HR(e.shape,this.elementShape,`TensorList shape mismatch: `),this.maxNumElements===this.size())throw Error(`Trying to push element into a full list.`);Uo(e),this.tensors.push(e)}resize(t){if(t<0)throw Error(`TensorListResize expects size to be non-negative. Got: ${t}`);if(this.maxNumElements!==-1&&t>this.maxNumElements)throw Error(`TensorListResize input size ${t} is greater maxNumElement ${this.maxNumElements}.`);let n=new e([],this.elementShape,this.elementDtype,this.maxNumElements);n.tensors.length=t;for(let e=0;e<Math.min(this.tensors.length,t);++e)n.tensors[e]=this.tensors[e];return n}getItem(e,t,n){if(n!==this.elementDtype)throw Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);if(e<0||e>this.tensors.length)throw Error(`Trying to access element ${e} in a list with ${this.tensors.length} elements.`);if(this.tensors[e]==null)throw Error(`element at index ${e} is null.`);HR(this.tensors[e].shape,t,`TensorList shape mismatch: `);let r=WR(this.elementShape,this.tensors,t);return V(this.tensors[e],r)}setItem(e,t){if(t.dtype!==this.elementDtype)throw Error(`Invalid data types; op elements ${t.dtype}, but list elements ${this.elementDtype}`);if(e<0||this.maxNumElements!==-1&&e>=this.maxNumElements)throw Error(`Trying to set element ${e} in a list with max ${this.maxNumElements} elements.`);HR(this.elementShape,t.shape,`TensorList shape mismatch: `),Uo(t),this.tensors[e]!=null&&(this.tensors[e].kept=!1),this.tensors[e]=t}gather(e,t,n){if(t!==this.elementDtype)throw Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);HR(this.elementShape,n,`TensorList shape mismatch: `),e=e.slice(0,this.size());let r=WR(this.elementShape,this.tensors,n);return e.length===0?Ki([],[0].concat(r)):Vo(()=>hm(e.map(e=>V(this.tensors[e],r)),0))}concat(e,t){if(e&&e!==this.elementDtype)throw Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${e}`);HR(this.elementShape,t,`TensorList shape mismatch: `);let n=WR(this.elementShape,this.tensors,t);return this.size()===0?Ki([],[0].concat(n)):Vo(()=>Pc(this.tensors.map(e=>V(e,n)),0))}};function JR(e,t,n){let r=e.dtype;if(e.shape.length<1)throw Error(`Tensor must be at least a vector, but saw shape: ${e.shape}`);if(e.dtype!==n)throw Error(`Invalid data types; op elements ${e.dtype}, but list elements ${n}`);return HR(e.shape.slice(1),t,`TensorList shape mismatch: `),new qR(Im(e),t,r)}function YR(e,t,n,r){return new qR([],e,t,r)}function XR(e,t,n,r){if(t.length!==e.shape[0])throw Error(`Expected len(indices) == tensor.shape[0], but saw: ${t.length} vs. ${e.shape[0]}`);let i=Math.max(...t);if(r!=null&&r!==-1&&i>=r)throw Error(`Max index must be < array size (${i}  vs. ${r})`);let a=new qR([],n,e.dtype,r),o=Im(e,0);return t.forEach((e,t)=>{a.setItem(e,o[t])}),a}function ZR(e,t,n){let r=0,i=t.map(e=>(r+=e,r));if(r!==e.shape[0])throw Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${r}, and tensor's shape is: ${e.shape}`);let a=GR(e.shape.slice(1),n),o=r===0?0:e.size/r,s=Vo(()=>{let n=[];e=V(e,[1,r,o]);for(let r=0;r<t.length;++r){let s=[0,r===0?0:i[r-1],0],c=[1,t[r],o];n[r]=V(H(e,s,c),a)}return e.dispose(),n}),c=new qR([],n,e.dtype,t.length);for(let e=0;e<s.length;e++)c.setItem(e,s[e]);return c}var QR=async(e,t,n)=>{switch(e.op){case`If`:case`StatelessIf`:{let r=$(`thenBranch`,e,t,n),i=$(`elseBranch`,e,t,n),a=$(`cond`,e,t,n),o=$(`args`,e,t,n);return(await a.data())[0]?n.functionMap[r].executeFunctionAsync(o,n.tensorArrayMap,n.tensorListMap):n.functionMap[i].executeFunctionAsync(o,n.tensorArrayMap,n.tensorListMap)}case`While`:case`StatelessWhile`:{let r=$(`body`,e,t,n),i=$(`cond`,e,t,n),a=$(`args`,e,t,n),o=await n.functionMap[i].executeFunctionAsync(a,n.tensorArrayMap,n.tensorListMap),s=a.map(e=>e.id),c=await o[0].data();o.forEach(e=>{!e.kept&&s.indexOf(e.id)===-1&&e.dispose()});let l=a;for(;c[0];){let e=l;l=await n.functionMap[r].executeFunctionAsync(l,n.tensorArrayMap,n.tensorListMap);let t=l.map(e=>e.id);e.forEach(e=>{!e.kept&&s.indexOf(e.id)===-1&&t.indexOf(e.id)===-1&&e.dispose()});let a=await n.functionMap[i].executeFunctionAsync(l,n.tensorArrayMap,n.tensorListMap);c=await a[0].data(),a.forEach(e=>{!e.kept&&s.indexOf(e.id)===-1&&t.indexOf(e.id)===-1&&e.dispose()})}return l}case`LoopCond`:return[LL($(`pred`,e,t,n))];case`Switch`:{let r=$(`pred`,e,t,n),i=$(`data`,e,t,n);return i.kept||(i=LL(i)),(await r.data())[0]?[void 0,i]:[i,void 0]}case`Merge`:{let r=e.inputNames.find(e=>jL(e,t,n)!==void 0);return r?[LL(jL(r,t,n))]:void 0}case`Enter`:{let r=$(`frameName`,e,t,n),i=$(`tensor`,e,t,n);return n.enterFrame(r),[LL(i)]}case`Exit`:{let r=$(`tensor`,e,t,n);return n.exitFrame(),[LL(r)]}case`NextIteration`:{let r=$(`tensor`,e,t,n);return n.nextIteration(),[LL(r)]}case`TensorArrayV3`:{let r=$(`size`,e,t,n),i=$(`dtype`,e,t,n),a=$(`elementShape`,e,t,n),o=$(`dynamicSize`,e,t,n),s=$(`clearAfterRead`,e,t,n),c=$(`identicalElementShapes`,e,t,n),l=new KR($(`name`,e,t,n),i,r,a,c,o,s);return n.addTensorArray(l),[l.idTensor,U(1)]}case`TensorArrayWriteV3`:{let r=$(`tensorArrayId`,e,t,n),i=$(`index`,e,t,n),a=$(`tensor`,e,t,n),o=n.getTensorArray(r.id);return o.write(i,a),[o.idTensor]}case`TensorArrayReadV3`:{let r=$(`tensorArrayId`,e,t,n),i=$(`index`,e,t,n);return[n.getTensorArray(r.id).read(i)]}case`TensorArrayGatherV3`:{let r=$(`tensorArrayId`,e,t,n),i=$(`indices`,e,t,n),a=$(`dtype`,e,t,n);return[n.getTensorArray(r.id).gather(i,a)]}case`TensorArrayScatterV3`:{let r=$(`tensorArrayId`,e,t,n),i=$(`indices`,e,t,n),a=$(`tensor`,e,t,n),o=n.getTensorArray(r.id);return o.scatter(i,a),[o.idTensor]}case`TensorArrayConcatV3`:{let r=$(`tensorArrayId`,e,t,n),i=n.getTensorArray(r.id),a=$(`dtype`,e,t,n);return[i.concat(a)]}case`TensorArraySplitV3`:{let r=$(`tensorArrayId`,e,t,n),i=$(`tensor`,e,t,n),a=$(`lengths`,e,t,n),o=n.getTensorArray(r.id);return o.split(a,i),[o.idTensor]}case`TensorArraySizeV3`:{let r=$(`tensorArrayId`,e,t,n);return[U(n.getTensorArray(r.id).size(),`int32`)]}case`TensorArrayCloseV3`:{let r=$(`tensorArrayId`,e,t,n),i=n.getTensorArray(r.id);return i.clearAndClose(),[i.idTensor]}case`TensorListSetItem`:{let r=$(`tensorListId`,e,t,n),i=$(`index`,e,t,n),a=$(`tensor`,e,t,n),o=n.getTensorList(r.id);return o.setItem(i,a),[o.idTensor]}case`TensorListGetItem`:{let r=$(`tensorListId`,e,t,n),i=$(`index`,e,t,n),a=$(`elementShape`,e,t,n),o=$(`elementDType`,e,t,n);return[n.getTensorList(r.id).getItem(i,a,o)]}case`TensorListScatterV2`:case`TensorListScatter`:{let r=$(`indices`,e,t,n),i=XR($(`tensor`,e,t,n),r,$(`elementShape`,e,t,n),$(`numElements`,e,t,n));return n.addTensorList(i),[i.idTensor]}case`TensorListReserve`:case`EmptyTensorList`:{let r=$(`elementShape`,e,t,n),i=$(`elementDType`,e,t,n),a;a=e.op===`TensorListReserve`?`numElements`:`maxNumElements`;let o=$(a,e,t,n),s=YR(r,i,o,e.op===`TensorListReserve`?-1:o);return n.addTensorList(s),[s.idTensor]}case`TensorListGather`:{let r=$(`tensorListId`,e,t,n),i=$(`indices`,e,t,n),a=$(`elementShape`,e,t,n),o=$(`elementDType`,e,t,n);return[n.getTensorList(r.id).gather(i,o,a)]}case`TensorListStack`:{let r=$(`tensorListId`,e,t,n),i=$(`elementShape`,e,t,n),a=$(`elementDType`,e,t,n),o=$(`numElements`,e,t,n);return[n.getTensorList(r.id).stack(i,a,o)]}case`TensorListFromTensor`:{let r=JR($(`tensor`,e,t,n),$(`elementShape`,e,t,n),$(`elementDType`,e,t,n));return n.addTensorList(r),[r.idTensor]}case`TensorListConcat`:case`TensorListConcatV2`:{let r=$(`tensorListId`,e,t,n),i=n.getTensorList(r.id),a=$(`dtype`,e,t,n),o=$(`elementShape`,e,t,n);return[i.concat(a,o)]}case`TensorListPushBack`:{let r=$(`tensorListId`,e,t,n),i=$(`tensor`,e,t,n),a=n.getTensorList(r.id);return a.pushBack(i),[a.idTensor]}case`TensorListPopBack`:{let r=$(`tensorListId`,e,t,n),i=$(`elementShape`,e,t,n),a=$(`elementDType`,e,t,n);return[n.getTensorList(r.id).popBack(i,a)]}case`TensorListSplit`:{let r=$(`tensor`,e,t,n),i=$(`elementShape`,e,t,n),a=ZR(r,$(`lengths`,e,t,n),i);return n.addTensorList(a),[a.idTensor]}case`TensorListLength`:{let r=$(`tensorListId`,e,t,n);return[U(n.getTensorList(r.id).size(),`int32`)]}case`TensorListResize`:{let r=$(`tensorListId`,e,t,n),i=$(`size`,e,t,n),a=n.getTensorList(r.id).resize(i);return n.addTensorList(a),[a.idTensor]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};function $R(e,t,n){let[r,i]=$(`fusedOps`,e,t,n),a=r===`biasadd`,o=!a,s=i===`prelu`,c=r===`fusedbatchnorm`,l=$(`numArgs`,e,t,n);if(a){if(s&&l!==2)throw Error(`FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu must have two extra arguments: bias and alpha.`);if(!s&&a&&l!==1)throw Error(`FusedConv2d and DepthwiseConv2d with BiasAdd must have one extra argument: bias.`)}if(c)throw Error(`FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported`);let u=$(`strides`,e,t,n),d=IL(e,t,n),f=$(`dataFormat`,e,t,n).toUpperCase(),p=$(`dilations`,e,t,n),[m,h]=$(`args`,e,t,n);o&&(h=m,m=void 0);let g=$(`leakyreluAlpha`,e,t,n);return{stride:u,pad:d,dataFormat:f,dilations:p,biasArg:m,preluArg:h,activationFunc:i,leakyreluAlpha:g}}var ez=(e,t,n,r=zR)=>{switch(e.op){case`Conv1D`:{let i=$(`stride`,e,t,n),a=$(`pad`,e,t,n),o=$(`dataFormat`,e,t,n).toUpperCase(),s=$(`dilation`,e,t,n);return[r.conv1d($(`x`,e,t,n),$(`filter`,e,t,n),i,a,o,s)]}case`Conv2D`:{let i=$(`strides`,e,t,n),a=IL(e,t,n),o=$(`dataFormat`,e,t,n).toUpperCase(),s=$(`dilations`,e,t,n);return[r.conv2d($(`x`,e,t,n),$(`filter`,e,t,n),[i[1],i[2]],a,o,[s[1],s[2]])]}case`_FusedConv2D`:{let{stride:i,pad:a,dataFormat:o,dilations:s,biasArg:c,preluArg:l,activationFunc:u,leakyreluAlpha:d}=$R(e,t,n);return[r.fused.conv2d({x:$(`x`,e,t,n),filter:$(`filter`,e,t,n),strides:[i[1],i[2]],pad:a,dataFormat:o,dilations:[s[1],s[2]],bias:c,activation:u,preluActivationWeights:l,leakyreluAlpha:d})]}case`FusedDepthwiseConv2dNative`:{let{stride:i,pad:a,dataFormat:o,dilations:s,biasArg:c,preluArg:l,activationFunc:u,leakyreluAlpha:d}=$R(e,t,n);return[r.fused.depthwiseConv2d({x:$(`x`,e,t,n),filter:$(`filter`,e,t,n),strides:[i[1],i[2]],pad:a,dataFormat:o,dilations:[s[1],s[2]],bias:c,activation:u,preluActivationWeights:l,leakyreluAlpha:d})]}case`Conv2DBackpropInput`:case`Conv2dTranspose`:{let i=$(`outputShape`,e,t,n),a=$(`strides`,e,t,n),o=IL(e,t,n);return[r.conv2dTranspose($(`x`,e,t,n),$(`filter`,e,t,n),i,[a[1],a[2]],o)]}case`DepthwiseConv2dNative`:case`DepthwiseConv2d`:{let i=$(`strides`,e,t,n),a=IL(e,t,n),o=$(`dilations`,e,t,n),s=$(`dataFormat`,e,t,n).toUpperCase();return[r.depthwiseConv2d($(`input`,e,t,n),$(`filter`,e,t,n),[i[1],i[2]],a,s,[o[1],o[2]])]}case`Conv3D`:{let i=$(`strides`,e,t,n),a=$(`pad`,e,t,n),o=$(`dataFormat`,e,t,n).toUpperCase(),s=$(`dilations`,e,t,n);return[r.conv3d($(`x`,e,t,n),$(`filter`,e,t,n),[i[1],i[2],i[3]],a,o,[s[1],s[2],s[3]])]}case`AvgPool`:{let i=$(`strides`,e,t,n),a=$(`pad`,e,t,n),o=$(`kernelSize`,e,t,n);return[r.avgPool($(`x`,e,t,n),[o[1],o[2]],[i[1],i[2]],a)]}case`MaxPool`:{let i=$(`strides`,e,t,n),a=$(`pad`,e,t,n),o=$(`kernelSize`,e,t,n);return[r.maxPool($(`x`,e,t,n),[o[1],o[2]],[i[1],i[2]],a)]}case`MaxPoolWithArgmax`:{let i=$(`strides`,e,t,n),a=$(`pad`,e,t,n),o=$(`kernelSize`,e,t,n),s=$(`includeBatchInIndex`,e,t,n),{result:c,indexes:l}=r.maxPoolWithArgmax($(`x`,e,t,n),[o[1],o[2]],[i[1],i[2]],a,s);return[c,l]}case`AvgPool3D`:{let i=$(`strides`,e,t,n),a=$(`pad`,e,t,n),o=$(`kernelSize`,e,t,n);return[r.avgPool3d($(`x`,e,t,n),[o[1],o[2],o[3]],[i[1],i[2],i[3]],a)]}case`MaxPool3D`:{let i=$(`strides`,e,t,n),a=$(`pad`,e,t,n),o=$(`kernelSize`,e,t,n);return[r.maxPool3d($(`x`,e,t,n),[o[1],o[2],o[3]],[i[1],i[2],i[3]],a)]}case`Dilation2D`:{let i=$(`strides`,e,t,n),a=$(`pad`,e,t,n),o=$(`dilations`,e,t,n),s=i[1],c=i[2],l=o[1],u=o[2];return[r.dilation2d($(`x`,e,t,n),$(`filter`,e,t,n),[s,c],a,[l,u],`NHWC`)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},tz=(e,t,n,r=zR)=>{switch(e.op){case`Fill`:{let i=$(`shape`,e,t,n),a=$(`dtype`,e,t,n),o=$(`value`,e,t,n);return[r.fill(i,o,a)]}case`LinSpace`:{let i=$(`start`,e,t,n),a=$(`stop`,e,t,n),o=$(`num`,e,t,n);return[r.linspace(i,a,o)]}case`Multinomial`:{let i=$(`logits`,e,t,n),a=$(`numSamples`,e,t,n),o=$(`seed`,e,t,n);return[r.multinomial(i,a,o)]}case`OneHot`:{let i=$(`indices`,e,t,n),a=$(`depth`,e,t,n),o=$(`onValue`,e,t,n),s=$(`offValue`,e,t,n),c=$(`dtype`,e,t,n);return[r.oneHot(i,a,o,s,c)]}case`Ones`:return[r.ones($(`shape`,e,t,n),$(`dtype`,e,t,n))];case`OnesLike`:return[r.onesLike($(`x`,e,t,n))];case`RandomStandardNormal`:return[r.randomStandardNormal($(`shape`,e,t,n),$(`dtype`,e,t,n),$(`seed`,e,t,n))];case`RandomUniform`:return[r.randomUniform($(`shape`,e,t,n),$(`minval`,e,t,n),$(`maxval`,e,t,n),$(`dtype`,e,t,n))];case`Range`:{let i=$(`start`,e,t,n),a=$(`stop`,e,t,n),o=$(`step`,e,t,n);return[r.range(i,a,o,$(`dtype`,e,t,n))]}case`TruncatedNormal`:{let i=$(`shape`,e,t,n),a=$(`mean`,e,t,n),o=$(`stdDev`,e,t,n),s=$(`seed`,e,t,n);return[r.truncatedNormal(i,a,o,$(`dtype`,e,t,n),s)]}case`Zeros`:return[r.zeros($(`shape`,e,t,n),$(`dtype`,e,t,n))];case`ZerosLike`:return[r.zerosLike($(`x`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};function nz(e,t,n){return{boxes:$(`boxes`,e,t,n),scores:$(`scores`,e,t,n),maxOutputSize:$(`maxOutputSize`,e,t,n),iouThreshold:$(`iouThreshold`,e,t,n),scoreThreshold:$(`scoreThreshold`,e,t,n),softNmsSigma:$(`softNmsSigma`,e,t,n)}}var rz=async(e,t,n,r,i=zR)=>{switch(e.op){case`NonMaxSuppressionV5`:{let{boxes:r,scores:a,maxOutputSize:o,iouThreshold:s,scoreThreshold:c,softNmsSigma:l}=nz(e,t,n),u=await i.image.nonMaxSuppressionWithScoreAsync(r,a,o,s,c,l);return[u.selectedIndices,u.selectedScores]}case`NonMaxSuppressionV4`:{let{boxes:r,scores:a,maxOutputSize:o,iouThreshold:s,scoreThreshold:c}=nz(e,t,n),l=$(`padToMaxOutputSize`,e,t,n),u=await i.image.nonMaxSuppressionPaddedAsync(r,a,o,s,c,l);return[u.selectedIndices,u.validOutputs]}case`NonMaxSuppressionV3`:case`NonMaxSuppressionV2`:{let{boxes:r,scores:a,maxOutputSize:o,iouThreshold:s,scoreThreshold:c}=nz(e,t,n);return[await i.image.nonMaxSuppressionAsync(r,a,o,s,c)]}case`Where`:{let r=i.cast($(`condition`,e,t,n),`bool`),a=[await i.whereAsync(r)];return r.dispose(),a}case`ListDiff`:return i.setdiff1dAsync($(`x`,e,t,n),$(`y`,e,t,n));default:throw TypeError(`Node type ${e.op} is not implemented`)}},iz=(e,t,n,r=zR)=>{switch(e.op){case`LowerBound`:{let i=$(`sortedSequence`,e,t,n),a=$(`values`,e,t,n);return[r.lowerBound(i,a)]}case`TopKV2`:{let i=$(`x`,e,t,n),a=$(`k`,e,t,n),o=$(`sorted`,e,t,n),s=r.topk(i,a,o);return[s.values,s.indices]}case`UpperBound`:{let i=$(`sortedSequence`,e,t,n),a=$(`values`,e,t,n);return[r.upperBound(i,a)]}case`Unique`:{let i=$(`x`,e,t,n),a=r.unique(i);return[a.values,a.indices]}case`UniqueV2`:{let i=$(`x`,e,t,n),a=$(`axis`,e,t,n),o=r.unique(i,a);return[o.values,o.indices]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},az=(e,t,n,r=zR)=>{switch(e.op){case`Const`:return t[e.name];case`PlaceholderWithDefault`:let i=$(`default`,e,t,n);return[jL(e.name,t,n)||i];case`Placeholder`:return[jL(e.name,t,n)];case`Identity`:case`StopGradient`:case`FakeQuantWithMinMaxVars`:return[LL($(`x`,e,t,n))];case`IdentityN`:return $(`x`,e,t,n).map(e=>LL(e));case`Snapshot`:return[LL($(`x`,e,t,n))];case`Shape`:return[r.tensor1d($(`x`,e,t,n).shape,`int32`)];case`ShapeN`:return $(`x`,e,t,n).map(e=>r.tensor1d(e.shape));case`Size`:return[r.scalar($(`x`,e,t,n).size,`int32`)];case`Rank`:return[r.scalar($(`x`,e,t,n).rank,`int32`)];case`NoOp`:return[r.scalar(1)];case`Print`:let a=$(`x`,e,t,n),o=$(`data`,e,t,n),s=$(`message`,e,t,n),c=$(`summarize`,e,t,n);console.warn(`The graph has a tf.print() operation,usually used for debugging, which slows down performance.`),console.log(s);for(let e=0;e<o.length;e++)console.log(Array.prototype.slice.call(o[e].dataSync()).slice(0,c));return[a];default:throw TypeError(`Node type ${e.op} is not implemented`)}},oz=class{constructor(e,t){this.keyDType=e,this.valueDType=t,this.handle=U(0),this.tensorMap=new Map,Uo(this.handle)}get id(){return this.handle.id}clearAndClose(){this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),this.handle.dispose()}size(){return this.tensorMap.size}tensorSize(){return U(this.size(),`int32`)}async import(e,t){this.checkKeyAndValueTensor(e,t);let n=await e.data();return this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),Vo(()=>{let e=Im(t),r=n.length,i=e.length;f(r===i,()=>`The number of elements doesn't match, keys has ${r} elements, the values has ${i} elements.`);for(let t=0;t<r;t++){let r=n[t],i=e[t];Uo(i),this.tensorMap.set(r,i)}return this.handle})}async find(e,t){this.checkKeyAndValueTensor(e,t);let n=await e.data();return Vo(()=>{let e=[];for(let r=0;r<n.length;r++){let i=n[r],a=this.findWithDefault(i,t);e.push(a)}return hm(e)})}findWithDefault(e,t){return this.tensorMap.get(e)??t}checkKeyAndValueTensor(e,t){if(e.dtype!==this.keyDType)throw Error(`Expect key dtype ${this.keyDType}, but got ${e.dtype}`);if(t.dtype!==this.valueDType)throw Error(`Expect value dtype ${this.valueDType}, but got ${t.dtype}`)}},sz=async(e,t,n,r)=>{switch(e.op){case`HashTable`:case`HashTableV2`:{let i=new oz($(`keyDType`,e,t,n),$(`valueDType`,e,t,n));return r.addHashTable(e.name,i),[i.handle]}case`LookupTableImport`:case`LookupTableImportV2`:{let i=$(`tableHandle`,e,t,n,r),a=$(`keys`,e,t,n),o=$(`values`,e,t,n);return[await r.getHashTableById(i.id).import(a,o)]}case`LookupTableFind`:case`LookupTableFindV2`:{let i=$(`tableHandle`,e,t,n,r),a=$(`keys`,e,t,n),o=$(`defaultValue`,e,t,n);return[await r.getHashTableById(i.id).find(a,o)]}case`LookupTableSize`:case`LookupTableSizeV2`:{let i=$(`tableHandle`,e,t,n,r);return[r.getHashTableById(i.id).tensorSize()]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},cz=(e,t,n,r=zR)=>{switch(e.op){case`ResizeBilinear`:{let i=$(`images`,e,t,n),a=$(`size`,e,t,n),o=$(`alignCorners`,e,t,n),s=$(`halfPixelCenters`,e,t,n);return[r.image.resizeBilinear(i,[a[0],a[1]],o,s)]}case`ResizeNearestNeighbor`:{let i=$(`images`,e,t,n),a=$(`size`,e,t,n),o=$(`alignCorners`,e,t,n),s=$(`halfPixelCenters`,e,t,n);return[r.image.resizeNearestNeighbor(i,[a[0],a[1]],o,s)]}case`CropAndResize`:{let i=$(`image`,e,t,n),a=$(`boxes`,e,t,n),o=$(`boxInd`,e,t,n),s=$(`cropSize`,e,t,n),c=$(`method`,e,t,n),l=$(`extrapolationValue`,e,t,n);return[r.image.cropAndResize(i,a,o,s,c,l)]}case`ImageProjectiveTransformV3`:{let i=$(`images`,e,t,n),a=$(`transforms`,e,t,n),o=$(`outputShape`,e,t,n),s=$(`fillValue`,e,t,n),c=$(`interpolation`,e,t,n),l=$(`fillMode`,e,t,n);return[r.image.transform(i,a,c.toLowerCase(),l.toLowerCase(),s,o)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},lz=(e,t,n,r=zR)=>{switch(e.op){case`Equal`:return[r.equal($(`a`,e,t,n),$(`b`,e,t,n))];case`NotEqual`:return[r.notEqual($(`a`,e,t,n),$(`b`,e,t,n))];case`Greater`:return[r.greater($(`a`,e,t,n),$(`b`,e,t,n))];case`GreaterEqual`:return[r.greaterEqual($(`a`,e,t,n),$(`b`,e,t,n))];case`Less`:return[r.less($(`a`,e,t,n),$(`b`,e,t,n))];case`LessEqual`:return[r.lessEqual($(`a`,e,t,n),$(`b`,e,t,n))];case`LogicalAnd`:return[r.logicalAnd($(`a`,e,t,n),$(`b`,e,t,n))];case`LogicalNot`:return[r.logicalNot($(`a`,e,t,n))];case`LogicalOr`:return[r.logicalOr($(`a`,e,t,n),$(`b`,e,t,n))];case`Select`:case`SelectV2`:return[r.where($(`condition`,e,t,n),$(`a`,e,t,n),$(`b`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},uz=(e,t,n,r=zR)=>{switch(e.op){case`BatchMatMul`:case`BatchMatMulV2`:case`MatMul`:return[r.matMul($(`a`,e,t,n),$(`b`,e,t,n),$(`transposeA`,e,t,n),$(`transposeB`,e,t,n))];case`Einsum`:return[r.einsum($(`equation`,e,t,n),...$(`tensors`,e,t,n))];case`Transpose`:return[r.transpose($(`x`,e,t,n),$(`perm`,e,t,n))];case`_FusedMatMul`:let[i,a]=$(`fusedOps`,e,t,n),o=i===`biasadd`,s=a===`prelu`,c=$(`numArgs`,e,t,n),l=$(`leakyreluAlpha`,e,t,n);if(o){if(s&&c!==2)throw Error(`Fused MatMul with BiasAdd and Prelu must have two extra arguments: bias and alpha.`);if(!s&&c!==1)throw Error(`Fused MatMul with BiasAdd must have one extra argument: bias.`)}let[u,d]=$(`args`,e,t,n);return[r.fused.matMul({a:$(`a`,e,t,n),b:$(`b`,e,t,n),transposeA:$(`transposeA`,e,t,n),transposeB:$(`transposeB`,e,t,n),bias:u,activation:a,preluActivationWeights:d,leakyreluAlpha:l})];default:throw TypeError(`Node type ${e.op} is not implemented`)}},dz=(e,t,n,r=zR)=>{switch(e.op){case`EuclideanNorm`:return[r.euclideanNorm($(`x`,e,t,n),$(`axis`,e,t,n),$(`keepDims`,e,t,n))];case`FusedBatchNorm`:case`FusedBatchNormV2`:return[r.batchNorm($(`x`,e,t,n),$(`mean`,e,t,n),$(`variance`,e,t,n),$(`offset`,e,t,n),$(`scale`,e,t,n),$(`epsilon`,e,t,n))];case`FusedBatchNormV3`:return[r.batchNorm($(`x`,e,t,n),$(`mean`,e,t,n),$(`variance`,e,t,n),$(`offset`,e,t,n),$(`scale`,e,t,n),$(`epsilon`,e,t,n))];case`LRN`:return[r.localResponseNormalization($(`x`,e,t,n),$(`radius`,e,t,n),$(`bias`,e,t,n),$(`alpha`,e,t,n),$(`beta`,e,t,n))];case`Softmax`:return[r.softmax($(`x`,e,t,n))];case`LogSoftmax`:return[r.logSoftmax($(`x`,e,t,n))];case`SparseToDense`:return[r.sparseToDense($(`sparseIndices`,e,t,n),$(`outputShape`,e,t,n),$(`sparseValues`,e,t,n),$(`defaultValue`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},fz=(e,t,n,r=zR)=>{switch(e.op){case`Max`:{let i=$(`axis`,e,t,n),a=$(`keepDims`,e,t,n);return[r.max($(`x`,e,t,n),i,a)]}case`Mean`:{let i=$(`axis`,e,t,n),a=$(`keepDims`,e,t,n);return[r.mean($(`x`,e,t,n),i,a)]}case`Min`:{let i=$(`axis`,e,t,n),a=$(`keepDims`,e,t,n);return[r.min($(`x`,e,t,n),i,a)]}case`Sum`:{let i=$(`axis`,e,t,n),a=$(`keepDims`,e,t,n);return[r.sum($(`x`,e,t,n),i,a)]}case`All`:{let i=$(`axis`,e,t,n),a=$(`keepDims`,e,t,n);return[r.all($(`x`,e,t,n),i,a)]}case`Any`:{let i=$(`axis`,e,t,n),a=$(`keepDims`,e,t,n);return[r.any($(`x`,e,t,n),i,a)]}case`ArgMax`:{let i=$(`axis`,e,t,n);return[r.argMax($(`x`,e,t,n),i)]}case`ArgMin`:{let i=$(`axis`,e,t,n);return[r.argMin($(`x`,e,t,n),i)]}case`Prod`:{let i=$(`axis`,e,t,n),a=$(`keepDims`,e,t,n);return[r.prod($(`x`,e,t,n),i,a)]}case`Cumprod`:{let i=$(`axis`,e,t,n),a=$(`exclusive`,e,t,n),o=$(`reverse`,e,t,n);return[r.cumprod($(`x`,e,t,n),i,a,o)]}case`Cumsum`:{let i=$(`axis`,e,t,n),a=$(`exclusive`,e,t,n),o=$(`reverse`,e,t,n);return[r.cumsum($(`x`,e,t,n),i,a,o)]}case`Bincount`:let i=$(`x`,e,t,n),a=$(`weights`,e,t,n),o=$(`size`,e,t,n);return[r.bincount(i,a,o)];case`DenseBincount`:{let i=$(`x`,e,t,n),a=$(`weights`,e,t,n),o=$(`size`,e,t,n),s=$(`binaryOutput`,e,t,n);return[r.denseBincount(i,a,o,s)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},pz=(e,t,n,r=zR)=>{switch(e.op){case`ConcatV2`:case`Concat`:{let i=$(`n`,e,t,n),a=$(`axis`,e,t,n),o=$(`tensors`,e,t,n);return o=o.slice(0,i),[r.concat(o,a)]}case`Gather`:{let i=$(`x`,e,t,n),a=$(`indices`,e,t,n);return[r.gather(i,r.cast(a,`int32`),0)]}case`GatherV2`:{let i=$(`axis`,e,t,n),a=$(`batchDims`,e,t,n),o=$(`x`,e,t,n),s=$(`indices`,e,t,n);return[r.gather(o,r.cast(s,`int32`),i,a)]}case`Reverse`:{let i=$(`dims`,e,t,n),a=[];for(let e=0;e<i.length;e++)i[e]&&a.push(e);let o=$(`x`,e,t,n);return[r.reverse(o,a)]}case`ReverseV2`:{let i=$(`axis`,e,t,n),a=$(`x`,e,t,n);return[r.reverse(a,i)]}case`Slice`:{let i=$(`begin`,e,t,n),a=$(`size`,e,t,n);return[r.slice($(`x`,e,t,n),i,a)]}case`StridedSlice`:{let i=$(`begin`,e,t,n),a=$(`end`,e,t,n),o=$(`strides`,e,t,n),s=$(`beginMask`,e,t,n),c=$(`endMask`,e,t,n),l=$(`ellipsisMask`,e,t,n),u=$(`newAxisMask`,e,t,n),d=$(`shrinkAxisMask`,e,t,n),f=$(`x`,e,t,n);return[r.stridedSlice(f,i,a,o,s,c,l,u,d)]}case`Pack`:return Vo(()=>{let i=$(`axis`,e,t,n),a=$(`tensors`,e,t,n),o=a[0].shape,s=r.squeeze(a[0]).shape,c=a.map(e=>{let t=_(e.shape,o);if(!t&&!_(r.squeeze(e).shape,s))throw Error(`the input tensors shape does not match`);return t?e:r.reshape(e,o)});return[r.stack(c,i)]});case`Unpack`:{let i=$(`axis`,e,t,n),a=$(`tensor`,e,t,n);return r.unstack(a,i)}case`Tile`:{let i=$(`reps`,e,t,n);return[r.tile($(`x`,e,t,n),i)]}case`Split`:case`SplitV`:{let i=$(`axis`,e,t,n),a=$(`numOrSizeSplits`,e,t,n),o=$(`x`,e,t,n);return r.split(o,a,i)}case`ScatterNd`:{let i=$(`indices`,e,t,n),a=$(`values`,e,t,n),o=$(`shape`,e,t,n);return[r.scatterND(i,a,o)]}case`GatherNd`:{let i=$(`x`,e,t,n),a=$(`indices`,e,t,n);return[r.gatherND(i,a)]}case`SparseToDense`:{let i=$(`sparseIndices`,e,t,n),a=$(`outputShape`,e,t,n),o=$(`sparseValues`,e,t,n),s=$(`defaultValue`,e,t,n);return[r.sparseToDense(i,o,a,o.dtype===s.dtype?s:r.cast(s,o.dtype))]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},mz=(e,t,n,r=zR)=>{switch(e.op){case`SparseFillEmptyRows`:{let{outputIndices:i,outputValues:a,emptyRowIndicator:o,reverseIndexMap:s}=r.sparse.sparseFillEmptyRows($(`indices`,e,t,n),$(`values`,e,t,n),$(`denseShape`,e,t,n),$(`defaultValue`,e,t,n));return[i,a,o,s]}case`SparseReshape`:{let{outputIndices:i,outputShape:a}=r.sparse.sparseReshape($(`inputIndices`,e,t,n),$(`inputShape`,e,t,n),$(`newShape`,e,t,n));return[i,a]}case`SparseSegmentMean`:return[r.sparse.sparseSegmentMean($(`data`,e,t,n),$(`indices`,e,t,n),$(`segmentIds`,e,t,n))];case`SparseSegmentSum`:return[r.sparse.sparseSegmentSum($(`data`,e,t,n),$(`indices`,e,t,n),$(`segmentIds`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},hz=(e,t,n,r=zR)=>{switch(e.op){case`FFT`:return[r.fft($(`x`,e,t,n))];case`IFFT`:return[r.ifft($(`x`,e,t,n))];case`RFFT`:return[r.rfft($(`x`,e,t,n))];case`IRFFT`:return[r.irfft($(`x`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},gz=(e,t,n,r=zR)=>{switch(e.op){case`StringNGrams`:{let{nGrams:i,nGramsSplits:a}=r.string.stringNGrams($(`data`,e,t,n),$(`dataSplits`,e,t,n),$(`separator`,e,t,n),$(`nGramWidths`,e,t,n),$(`leftPad`,e,t,n),$(`rightPad`,e,t,n),$(`padWidth`,e,t,n),$(`preserveShortSequences`,e,t,n));return[i,a]}case`StringSplit`:{let{indices:i,values:a,shape:o}=r.string.stringSplit($(`input`,e,t,n),$(`delimiter`,e,t,n),$(`skipEmpty`,e,t,n));return[i,a,o]}case`StringToHashBucketFast`:return[r.string.stringToHashBucketFast($(`input`,e,t,n),$(`numBuckets`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},_z=(e,t,n,r=zR)=>{switch(e.op){case`Cast`:return[r.cast($(`x`,e,t,n),$(`dtype`,e,t,n))];case`ExpandDims`:{let i=$(`axis`,e,t,n);return[r.expandDims($(`x`,e,t,n),i)]}case`Squeeze`:{let i=$(`axis`,e,t,n);return[r.squeeze($(`x`,e,t,n),i)]}case`Reshape`:return[r.reshape($(`x`,e,t,n),$(`shape`,e,t,n))];case`MirrorPad`:return[r.mirrorPad($(`x`,e,t,n),$(`padding`,e,t,n),$(`mode`,e,t,n))];case`PadV2`:case`Pad`:return[r.pad($(`x`,e,t,n),$(`padding`,e,t,n),$(`constantValue`,e,t,n))];case`SpaceToBatchND`:{let i=$(`blockShape`,e,t,n),a=$(`paddings`,e,t,n);return[r.spaceToBatchND($(`x`,e,t,n),i,a)]}case`BatchToSpaceND`:{let i=$(`blockShape`,e,t,n),a=$(`crops`,e,t,n);return[r.batchToSpaceND($(`x`,e,t,n),i,a)]}case`DepthToSpace`:{let i=$(`blockSize`,e,t,n),a=$(`dataFormat`,e,t,n).toUpperCase();return[r.depthToSpace($(`x`,e,t,n),i,a)]}case`BroadcastTo`:return[r.broadcastTo($(`x`,e,t,n),$(`shape`,e,t,n))];case`BroadcastArgs`:return[r.broadcastArgs($(`s0`,e,t,n),$(`s1`,e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};function vz(e,t,n,r,i=Vo){let a=((e,t,n)=>{switch(e.category){case`arithmetic`:return i(()=>BR(e,t,n));case`basic_math`:return i(()=>VR(e,t,n));case`control`:return QR(e,t,n);case`convolution`:return i(()=>ez(e,t,n));case`creation`:return i(()=>tz(e,t,n));case`dynamic`:return rz(e,t,n);case`evaluation`:return i(()=>iz(e,t,n));case`image`:return i(()=>cz(e,t,n));case`graph`:return i(()=>az(e,t,n));case`logical`:return i(()=>lz(e,t,n));case`matrices`:return i(()=>uz(e,t,n));case`normalization`:return i(()=>dz(e,t,n));case`reduction`:return i(()=>fz(e,t,n));case`slice_join`:return i(()=>pz(e,t,n));case`sparse`:return i(()=>mz(e,t,n));case`spectral`:return i(()=>hz(e,t,n));case`string`:return i(()=>gz(e,t,n));case`transformation`:return i(()=>_z(e,t,n));case`hash_table`:return sz(e,t,n,r);case`custom`:let a=AL(e.op);if(a&&a.customExecutor)return a.customExecutor(new RR(e,t,n));throw TypeError(`Custom op ${e.op} is not registered.`);default:throw TypeError(`Unknown op '${e.op}'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()`)}})(e,t,n);return he(a)?a.then(e=>[].concat(e)):[].concat(a)}var yz=class{constructor(e={},t={},n={},r={}){this.weightMap=e,this.tensorArrayMap=t,this.tensorListMap=n,this.functionMap=r,this.rootContext={id:0,frameName:``,iterationId:0},this.contexts=[this.rootContext],this.lastId=0,this.generateCurrentContextIds()}newFrame(e,t){return{id:e,frameName:t,iterationId:0}}set currentContext(e){this.contexts!==e&&(this.contexts=e,this.generateCurrentContextIds())}get currentContext(){return this.contexts}get currentContextId(){return this._currentContextIds[0]}get currentContextIds(){return this._currentContextIds}generateCurrentContextIds(){let e=[];for(let t=0;t<this.contexts.length-1;t++){let n=this.contexts.slice(0,this.contexts.length-t);e.push(this.contextIdforContexts(n))}e.push(``),this._currentContextIds=e}contextIdforContexts(e){return e?e.map(e=>e.id===0&&e.iterationId===0?``:`${e.frameName}-${e.iterationId}`).join(`/`):``}enterFrame(e){this.contexts&&(this.lastId++,this.contexts=this.contexts.slice(),this.contexts.push(this.newFrame(this.lastId,e)),this._currentContextIds.unshift(this.contextIdforContexts(this.contexts)))}exitFrame(){if(this.contexts&&this.contexts.length>1)this.contexts=this.contexts.slice(),this.contexts.splice(-1),this.currentContextIds.shift();else throw Error(`Cannot exit frame, the context is empty`)}nextIteration(){if(this.contexts&&this.contexts.length>0){this.contexts=this.contexts.slice(),this.lastId++;let e=Object.assign({},this.contexts[this.contexts.length-1]);e.iterationId+=1,e.id=this.lastId,this.contexts.splice(-1,1,e),this._currentContextIds.splice(0,1,this.contextIdforContexts(this.contexts))}else throw Error(`Cannot increase frame iteration, the context is empty`)}getWeight(e){return this.weightMap[e]}addTensorArray(e){this.tensorArrayMap[e.id]=e}getTensorArray(e){return this.tensorArrayMap[e]}addTensorList(e){this.tensorListMap[e.id]=e}getTensorList(e){return this.tensorListMap[e]}dispose(e){for(let t in this.tensorArrayMap)this.tensorArrayMap[t].clearAndClose(e);for(let t in this.tensorListMap)this.tensorListMap[t].clearAndClose(e)}};function bz(e,t,n,r){let i=new Set,a=[],o=null,s=null,c=new Set,l=Object.keys(e).map(e=>FL(e)[0]),u=[];r!=null&&(u=r.map(e=>FL(e.name)[0]));let d=[...t];for(;d.length>0;){let e=d.pop();(Tz(e)||Ez(e)||Dz(e))&&(o??(o=e,s=o.children.map(e=>e.name).filter(e=>i.has(e)))),i.add(e.name),n[e.name]==null&&l.indexOf(e.name)===-1&&u.indexOf(e.name)===-1&&(e.inputs.length===0?a.push(e.name):e.inputs.forEach(e=>{c.has(e.name)||(c.add(e.name),d.push(e))}))}return{inputs:e,outputs:t,usedNodes:i,missingInputs:a,dynamicNode:o,syncInputs:s}}function xz(e,t,n){let{usedNodes:r,inputs:i}=n,a=[],o=Object.keys(i).map(e=>FL(e)[0]).map(t=>e.nodes[t]),s=e.initNodes;o.forEach(e=>{r.has(e.name)&&a.push(e)}),e.weights.forEach(e=>{r.has(e.name)&&a.push(e)}),s?.forEach(e=>{r.has(e.name)&&a.push(e)});let c=new Set,l=[];for(;a.length>0;){let e=a.pop();c.add(e.name),t[e.name]||l.push(e),e.children.forEach(e=>{!c.has(e.name)&&r.has(e.name)&&e.inputs.every(e=>c.has(e.name))&&a.push(e)})}return l}var Sz=[`Switch`,`Merge`,`Enter`,`Exit`,`NextIteration`,`StatelessIf`,`StatelessWhile`,`if`,`While`],Cz=[`NonMaxSuppressionV2`,`NonMaxSuppressionV3`,`NonMaxSuppressionV5`,`Where`],wz=[`HashTable`,`HashTableV2`,`LookupTableImport`,`LookupTableImportV2`,`LookupTableFind`,`LookupTableFindV2`,`LookupTableSize`,`LookupTableSizeV2`];function Tz(e){return Sz.indexOf(e.op)>=0}function Ez(e){return Cz.indexOf(e.op)>=0}function Dz(e){return wz.indexOf(e.op)>=0}var Oz=class e{constructor(t,n){this.graph=t,this.parent=n,this.compiledMap=new Map,this._weightMap={},this.SEPERATOR=`,`,this._functions={},this._functionExecutorMap={},this.intermediateTensors={},this.keepTensorForDebug=!1,this._outputs=t.outputs,this._inputs=t.inputs,this._initNodes=t.initNodes,this._signature=t.signature,this._functions=t.functions,t.functions!=null&&Object.keys(t.functions).forEach(n=>{this._functionExecutorMap[n]=new e(t.functions[n],this)})}get weightIds(){return this.parent?this.parent.weightIds:this._weightIds}get functionExecutorMap(){return this.parent?this.parent.functionExecutorMap:this._functionExecutorMap}get weightMap(){return this.parent?this.parent.weightMap:this._weightMap}set weightMap(e){let t=Object.keys(e).map(t=>e[t].map(e=>e.id));this._weightIds=[].concat(...t),this._weightMap=e}set resourceManager(e){this._resourceManager=e}get inputs(){return this._inputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get outputs(){return this._outputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get inputNodes(){return this._inputs.map(e=>e.signatureKey||e.name)}get outputNodes(){return this._outputs.map(e=>{let t=e.signatureKey||e.name;return e.defaultOutput?`${t}:${e.defaultOutput}`:t})}get functions(){return Object.keys(this._functions).reduce((e,t)=>(e[t]=this._functions[t].signature,e),{})}getCompilationKey(e,t){let n=e.map(e=>e.name).sort(),r=t.map(e=>e.name).sort();return n.join(this.SEPERATOR)+`--`+r.join(this.SEPERATOR)}compile(e,t){let n=bz(e,t,this.weightMap,this._initNodes),{missingInputs:r,dynamicNode:i,syncInputs:a}=n;if(i!=null)throw Error(`This execution contains the node '${i.name}', which has the dynamic op '${i.op}'. Please use model.executeAsync() instead. Alternatively, to avoid the dynamic ops, specify the inputs [${a}]`);if(r.length>0){let n=t.map(e=>e.name),i=Object.keys(e);throw Error(`Cannot compute the outputs [${n}] from the provided inputs [${i}]. Missing the following inputs: [${r}]`)}return xz(this.graph,this.weightMap,n)}execute(e,t){e=this.mapInputs(e);let n=Object.keys(e).sort();this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t);let r=n.map(e=>this.graph.nodes[FL(e)[0]]),i=t.map(e=>FL(e)[0]),a=i.map(e=>this.graph.nodes[e]);this.resetIntermediateTensors(),a.length===0&&(a=this._outputs);let o=this.getCompilationKey(r,a),s=this.compiledMap.get(o);s??(s=this.compile(e,a),this.compiledMap.set(o,s));let c={},l={};return Vo(()=>{let n=new yz(this.weightMap,c,l,this.functionExecutorMap),r=Object.assign({},this.weightMap);Object.keys(e).forEach(t=>{let[n,i]=FL(t),a=[];a[i]=e[t],r[n]=a});let a=this.getFrozenTensorIds(r),o={};for(let e=0;e<s.length;e++){let t=s[e];if(!r[t.name]){let e=vz(t,r,n,this._resourceManager);if(he(e))throw Error(`The execution of the op '${t.op}' returned a promise. Please use model.executeAsync() instead.`);r[t.name]=e,this.checkTensorForDisposal(t.name,t,r,n,a,i,o)}}return this.parent??n.dispose(a),t.map(e=>jL(e,r,n))})}getFrozenTensorIds(e){let t=[].concat.apply([],Object.keys(e).map(t=>e[t]).map(e=>e.map(e=>e.id)));return new Set(t)}checkTensorForDisposal(e,t,n,r,i,a,o){t.category!==`control`&&a.indexOf(e)===-1&&(n[e].forEach(e=>{e!=null&&(o[e.id]=(o[e.id]||0)+t.children.length)}),t.inputs.forEach(e=>{e.category!==`control`&&ML(e.name,n,r)?.forEach(e=>{if(e&&!e.kept&&!i.has(e.id)){let n=o[e.id];if(n===1){if(!this.keepTensorForDebug)e.dispose();else{let[n,i]=NL(t.name,r);this.intermediateTensors[n]||(this.intermediateTensors[n]=[]),this.intermediateTensors[n][i]=e}delete o[e.id]}else n!=null&&o[e.id]--}})}))}async executeAsync(e,t){return this._executeAsync(e,t)}disposeIntermediateTensors(){this.intermediateTensors&&(Object.keys(this.intermediateTensors).forEach(e=>this.intermediateTensors[e].forEach(e=>e.dispose())),this.disposeTensorsMap())}disposeTensorsMap(){this.tensorsMap&&Object.keys(this.tensorsMap).forEach(e=>{this.tensorsMap[e].forEach(e=>{e&&!e.kept&&!e.isDisposed&&!this.keepIds.has(e.id)&&e.dispose()})})}getIntermediateTensors(){return this.tensorsMap}resetIntermediateTensors(){for(let e in this.intermediateTensors)this.intermediateTensors[e].forEach(e=>e.dispose()),delete this.intermediateTensors[e]}async _executeAsync(e,t,n=!1,r={},i={}){n||(e=this.mapInputs(e),this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t));try{this.keepTensorForDebug=N().getBool(`KEEP_INTERMEDIATE_TENSORS`)}catch(e){console.warn(e.message)}this.resetIntermediateTensors();let a=new yz(this.weightMap,r,i,this.functionExecutorMap);this.tensorsMap=await this.executeWithControlFlow(e,a,t,n);let o=t.map(e=>jL(e,this.tensorsMap,a)),s=o.map(e=>e.id),c=Object.keys(e).map(t=>e[t].id);return this.keepIds=new Set([...s,...c,...this.weightIds]),this.keepTensorForDebug||this.disposeTensorsMap(),this.parent??a.dispose(this.keepIds),o}async executeFunctionAsync(e,t,n){let r=e.reduce((e,t,n)=>(e[this.inputs[n].name]=t,e),{});return this._executeAsync(r,this.outputNodes,!0,t,n)}async executeWithControlFlow(e,t,n,r){let i=Object.keys(e),a=i.map(e=>this.graph.nodes[FL(e)[0]]),o=n.map(e=>FL(e)[0]),s=o.map(e=>this.graph.nodes[e]);s.length===0&&(s=this._outputs);let{usedNodes:c,missingInputs:l,dynamicNode:u,syncInputs:d}=bz(e,s,this.weightMap,this._initNodes),f=[...a,...this.graph.weights,...this._initNodes||[]].map(e=>({node:e,contexts:t.currentContext})),p=Object.assign({},this.weightMap);Object.keys(e).forEach(t=>{let[n,r]=FL(t),i=[];i[r]=e[t],p[n]=i});let m={},h=this.getFrozenTensorIds(p),g={};for(;f.length>0;){let e=this.processStack(a,f,t,p,g,h,o,m,c);await Promise.all(e)}u==null&&!r&&console.warn(`This model execution did not contain any nodes with control flow or dynamic output shapes. You can use model.execute() instead.`);let _=s.filter(e=>!Tz(e)&&!jL(e.name,p,t)).map(e=>e.name);if(_.length>0){let e=``;throw u!=null&&(e=`Alternatively, to avoid the dynamic ops, use model.execute() and specify the inputs [${d}]`),Error(`Cannot compute the outputs [${_}] from the provided inputs [${i}]. Consider providing the following inputs: [${l}]. ${e}`)}return p}processStack(e,t,n,r,i,a,o,s,c){let l=[];for(;t.length>0;){let e=t.pop();n.currentContext=e.contexts;let u=``;if(e.node.op===`Enter`&&$(`isConstant`,e.node,r,n)&&([u]=NL(e.node.name,n)),r[e.node.name]==null){let d=vz(e.node,r,n,this._resourceManager);u||([u]=NL(e.node.name,n));let f=n.currentContext;he(d)?l.push(d.then(l=>(r[u]=l,n.currentContext=f,this.checkTensorForDisposal(u,e.node,r,n,a,o,s),this.processChildNodes(e.node,t,n,r,i,c),l))):(r[u]=d,this.checkTensorForDisposal(u,e.node,r,n,a,o,s),this.processChildNodes(e.node,t,n,r,i,c))}else this.processChildNodes(e.node,t,n,r,i,c)}return l}processChildNodes(e,t,n,r,i,a){e.children.forEach(e=>{let[o]=NL(e.name,n);!i[o]&&a.has(e.name)&&(e.op===`Merge`?e.inputNames.some(e=>!!jL(e,r,n))&&(i[o]=!0,t.push({contexts:n.currentContext,node:e})):e.inputNames.every(e=>!!jL(e,r,n))&&(i[o]=!0,t.push({contexts:n.currentContext,node:e})))})}dispose(){Object.keys(this.weightMap).forEach(e=>this.weightMap[e].forEach(e=>e.dispose()))}checkInputShapeAndType(e){Object.keys(e).forEach(t=>{let n=e[t],[r]=FL(t),i=this.graph.nodes[r];if(i.attrParams.shape&&i.attrParams.shape.value){let e=i.attrParams.shape.value;f(e.length===n.shape.length&&n.shape.every((t,n)=>e[n]===-1||e[n]===t),()=>`The shape of dict['${i.name}'] provided in model.execute(dict) must be [${e}], but was [${n.shape}]`)}i.attrParams.dtype&&i.attrParams.dtype.value&&f(n.dtype===i.attrParams.dtype.value,()=>`The dtype of dict['${i.name}'] provided in model.execute(dict) must be ${i.attrParams.dtype.value}, but was ${n.dtype}`)})}mapInputs(e){let t={};for(let n in e)if(this._signature!=null&&this._signature.inputs!=null&&this._signature.inputs[n]!=null){let r=this._signature.inputs[n];t[r.name]=e[n]}else t[n]=e[n];return t}checkInputs(e){let t=Object.keys(e).filter(e=>{let[t]=FL(e);return this.graph.nodes[t]==null});if(t.length>0)throw Error(`The dict provided in model.execute(dict) has keys: [${t}] that are not part of graph`)}mapOutputs(e){return e.map(e=>this._signature!=null&&this._signature.outputs!=null&&this._signature.outputs[e]!=null?this._signature.outputs[e].name:e,{})}checkOutputs(e){e.forEach(e=>{let[t]=FL(e);if(!this.graph.nodes[t])throw Error(`The output '${e}' is not found in the graph`)})}},kz=class{constructor(e={},t={}){this.hashTableNameToHandle=e,this.hashTableMap=t}addHashTable(e,t){this.hashTableNameToHandle[e]=t.handle,this.hashTableMap[t.id]=t}getHashTableHandleByName(e){return this.hashTableNameToHandle[e]}getHashTableById(e){return this.hashTableMap[e]}dispose(){for(let e in this.hashTableMap)this.hashTableMap[e].clearAndClose(),delete this.hashTableMap[e];for(let e in this.hashTableNameToHandle)this.hashTableNameToHandle[e].dispose(),delete this.hashTableNameToHandle[e]}},Az=`?tfjs-format=file`,jz=`model.json`,Mz=class{constructor(e,t={},n=Io){this.modelUrl=e,this.loadOptions=t,this.version=`n/a`,this.io=n,t??(this.loadOptions={}),this.resourceManager=new kz}get modelVersion(){return this.version}get inputNodes(){return this.executor.inputNodes}get outputNodes(){return this.executor.outputNodes}get inputs(){return this.executor.inputs}get outputs(){return this.executor.outputs}get weights(){return this.executor.weightMap}get metadata(){return this.artifacts.userDefinedMetadata}get modelSignature(){return this.signature}get modelStructuredOutputKeys(){return this.structuredOutputKeys}findIOHandler(){let e=this.modelUrl;if(e.load!=null)this.handler=e;else if(this.loadOptions.requestInit!=null)this.handler=this.io.browserHTTPRequest(e,this.loadOptions);else{let t=this.io.getLoadHandlers(e,this.loadOptions);if(t.length===0)t.push(this.io.browserHTTPRequest(e,this.loadOptions));else if(t.length>1)throw Error(`Found more than one (${t.length}) load handlers for URL '${[e]}'`);this.handler=t[0]}}load(){if(this.findIOHandler(),this.handler.load==null)throw Error("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");let e=this.handler.load();return he(e)?e.then(e=>this.loadSync(e)):this.loadSync(e)}loadSync(e){this.artifacts=e;let t=this.artifacts.modelTopology,n=this.artifacts.signature;if(this.artifacts.userDefinedMetadata!=null){let e=this.artifacts.userDefinedMetadata;e.signature!=null&&(n=e.signature),e.structuredOutputKeys!=null&&(this.structuredOutputKeys=e.structuredOutputKeys)}this.signature=n,this.version=`${t.versions.producer}.${t.versions.minConsumer}`;let r=this.io.decodeWeights(this.artifacts.weightData,this.artifacts.weightSpecs);if(this.executor=new Oz(SR.Instance.transformGraph(t,this.signature)),this.executor.weightMap=this.convertTensorMapToTensorsMap(r),this.executor.resourceManager=this.resourceManager,e.modelInitializer!=null&&e.modelInitializer.node!=null){let t=SR.Instance.transformGraph(e.modelInitializer);this.initializer=new Oz(t),this.initializer.weightMap=this.executor.weightMap,this.initializer.resourceManager=this.resourceManager,this.initializer.executeAsync({},[])}return!0}async save(e,t){if(typeof e==`string`){let t=this.io.getSaveHandlers(e);if(t.length===0)throw Error(`Cannot find any save handlers for URL '${e}'`);if(t.length>1)throw Error(`Found more than one (${t.length}) save handlers for URL '${e}'`);e=t[0]}if(e.save==null)throw Error("GraphModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");return e.save(this.artifacts)}predict(e,t){let n=this.execute(e,this.outputNodes);if(this.structuredOutputKeys){let e=n instanceof fi?[n]:n,t={};return e.forEach((e,n)=>t[this.structuredOutputKeys[n]]=e),t}return n}normalizeInputs(e){if(!(e instanceof fi)&&!Array.isArray(e))return e;if(e=Array.isArray(e)?e:[e],e.length!==this.inputNodes.length)throw Error(`Input tensor count mismatch,the graph model has ${this.inputNodes.length} placeholders, while there are ${e.length} input tensors.`);return this.inputNodes.reduce((t,n,r)=>(t[n]=e[r],t),{})}normalizeOutputs(e){return e||=this.outputNodes,Array.isArray(e)?e:[e]}execute(e,t){e=this.normalizeInputs(e),t=this.normalizeOutputs(t);let n=this.executor.execute(e,t);return n.length>1?n:n[0]}async executeAsync(e,t){e=this.normalizeInputs(e),t=this.normalizeOutputs(t);let n=await this.executor.executeAsync(e,t);return n.length>1?n:n[0]}getIntermediateTensors(){return this.executor.getIntermediateTensors()}disposeIntermediateTensors(){this.executor.disposeIntermediateTensors()}convertTensorMapToTensorsMap(e){return Object.keys(e).reduce((t,n)=>(t[n]=[e[n]],t),{})}dispose(){this.executor.dispose(),this.initializer&&this.initializer.dispose(),this.resourceManager.dispose()}};async function Nz(e,t={},n=Io){if(e==null)throw Error(`modelUrl in loadGraphModel() cannot be null. Please provide a url or an IOHandler that loads the model`);t??={},t.fromTFHub&&typeof e==`string`&&(e=Pz(e));let r=new Mz(e,t,n);return await r.load(),r}function Pz(e){return e.endsWith(`/`)||(e+=`/`),`${e}${jz}${Az}`}export{Ko as a,N as c,qo as i,Sm as n,Go as o,as as r,Wo as s,Nz as t};