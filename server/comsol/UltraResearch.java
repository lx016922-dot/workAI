// COMSOL 6.4 research model: Cartesian 2D P-SV/SH, cylindrical axisymmetric
// elasticity and one-way thermoelasticity. Numeric input only; no supplied expressions.
import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class UltraResearch {
    private static Properties p, receipt;

    // WORKAI_MURNAGHAN_TOTAL_LAGRANGIAN_V1. Reference coordinates and dead
    // nominal traction; plane strain in z, with an optional anti-plane w field.
    private static boolean nonlinear=false;
    private static final String NL_SCHEMA="workai.murnaghan.guided_plate.scene.v1";
    private static final String[] CONTROLS={"linear_A","geometric_A","material_A","geometric_2A","material_2A"};
    private static String sha(String s)throws Exception{
        byte[] bytes=java.security.MessageDigest.getInstance("SHA-256").digest(s.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        StringBuilder out=new StringBuilder();for(byte b:bytes)out.append(String.format(Locale.ROOT,"%02x",b&255));return out.toString();
    }
    // Bounded JSON reader for the signed request: duplicate keys, nonfinite
    // numbers, invalid escapes/trailing content and excess nesting are rejected.
        private static String s; private static int at;
        private static void bad(){throw new IllegalArgumentException("Invalid request JSON");}
        private static void ws(){while(at<s.length()&&" \r\n\t".indexOf(s.charAt(at))>=0)at++;}
        private static char take(){if(at>=s.length())bad();return s.charAt(at++);}
        private static String str(){if(take()!='"')bad();StringBuilder b=new StringBuilder();while(true){char c=take();if(c=='"')return b.toString();if(c<32)bad();if(c=='\\'){c=take();switch(c){case '"':case '\\':case '/':b.append(c);break;case 'b':b.append('\b');break;case 'f':b.append('\f');break;case 'n':b.append('\n');break;case 'r':b.append('\r');break;case 't':b.append('\t');break;case 'u':if(at+4>s.length())bad();try{b.append((char)Integer.parseInt(s.substring(at,at+4),16));}catch(Exception e){bad();}at+=4;break;default:bad();}}else b.append(c);}}
        private static Object val(int depth){if(depth>12)bad();ws();if(at>=s.length())bad();char c=s.charAt(at);
            if(c=='"')return str();
            if(c=='{'){at++;Map<String,Object> m=new LinkedHashMap<>();ws();if(at<s.length()&&s.charAt(at)=='}'){at++;return m;}while(true){ws();String k=str();ws();if(take()!=':'||m.containsKey(k))bad();m.put(k,val(depth+1));ws();char end=take();if(end=='}')return m;if(end!=',')bad();}}
            if(c=='['){at++;List<Object> a=new ArrayList<>();ws();if(at<s.length()&&s.charAt(at)==']'){at++;return a;}while(true){a.add(val(depth+1));ws();char end=take();if(end==']')return a;if(end!=',')bad();}}
            if(s.startsWith("true",at)){at+=4;return Boolean.TRUE;}if(s.startsWith("false",at)){at+=5;return Boolean.FALSE;}if(s.startsWith("null",at)){at+=4;return null;}
            int start=at;while(at<s.length()&&"-+0123456789.eE".indexOf(s.charAt(at))>=0)at++;String tok=s.substring(start,at);
            if(!tok.matches("-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?"))bad();double v=Double.parseDouble(tok);if(!Double.isFinite(v))bad();return v;
        }
        private static Object parseJson(String input){s=input;at=0;if(s.length()>65536)bad();Object v=val(0);ws();if(at!=s.length())bad();return v;}
    @SuppressWarnings("unchecked") private static Map<String,Object> obj(Object v){if(!(v instanceof Map))throw new IllegalArgumentException("JSON object required");return (Map<String,Object>)v;}
    private static void keys(Map<String,Object> v,String... k){if(!v.keySet().equals(new HashSet<String>(Arrays.asList(k))))throw new IllegalArgumentException("Scene fields");}
    private static void eq(Object a,Object b){if(!Objects.equals(a,b))throw new IllegalArgumentException("Scene identity/value mismatch");}
    private static double num(Map<String,Object> v,String k){Object x=v.get(k);if(!(x instanceof Double))throw new IllegalArgumentException("Numeric scene value required");return (Double)x;}
    private static void put(Map<String,Object> v,String k,Object x){if(x==null)throw new IllegalArgumentException("Missing scene field");v.put(k,x);}
    private static void readScene()throws Exception{
        Map<String,Object> request=obj(parseJson(new String(Files.readAllBytes(Paths.get("request.json")),java.nio.charset.StandardCharsets.UTF_8)));
        for(String k:new String[]{"job_id","source_sha256","scene_sha256","parameters_sha256"})eq(request.get(k),p.getProperty(k));
        String sj=(String)request.get("scene_json"),pj=(String)request.get("parameters_json");eq(sha(sj),p.getProperty("scene_sha256"));eq(sha(pj),p.getProperty("parameters_sha256"));
        Map<String,Object> s=obj(parseJson(sj));if(!NL_SCHEMA.equals(s.get("schema")))return;
        nonlinear=true;
        keys(s,"schema","constitutive","geometry","material","material_reference","boundary","source","receivers","acquisition","numerics","nonlinear");
        eq(s.get("constitutive"),"murnaghan_green_lagrange");eq(s.get("boundary"),"free");
        Map<String,Object> ref=obj(s.get("material_reference"));keys(ref,"id","record_sha256","usage");
        eq(ref.get("id"),"aluminum-polycrystal-blaschke2017-v1");eq(ref.get("record_sha256"),"39102235cd2fae341a277e46aa08573ae2294710c79abd104b10491fe5edef0a");eq(ref.get("usage"),"murnaghan_total_lagrangian");
        Map<String,Object> law=obj(s.get("nonlinear"));keys(law,"definition","lambda_pa","mu_pa","l_pa","m_pa","n_pa","controls");
        eq(law.get("definition"),"murnaghan_green_lagrange_reference_volume");eq(law.get("controls"),Arrays.asList(CONTROLS));
        String[] ks={"lambda_pa","mu_pa","l_pa","m_pa","n_pa"};double[] vs={58.1e9,26.1e9,-143e9,-297e9,-345e9};for(int j=0;j<ks.length;j++)if(num(law,ks[j])!=vs[j])throw new IllegalArgumentException("Unreviewed material constants");
        Map<String,Object> g=obj(s.get("geometry")),mat=obj(s.get("material")),acq=obj(s.get("acquisition")),disc=obj(s.get("numerics")),tx=obj(s.get("source"));
        keys(g,"length_mm","thickness_mm");keys(mat,"rho_kg_m3","cl_m_s","ct_m_s");keys(acq,"duration_us","sample_step_us");
        keys(disc,"coarse_h_mm","fine_h_mm","max_step_us","relative_tolerance","field_nx","field_ny","field_frames");keys(tx,"profile","frequency_mhz","amplitude_pa","cycles","delay_us","phase_deg");eq(tx.get("profile"),"left_edge_uniform");
        double rho=num(mat,"rho_kg_m3"),cl=num(mat,"cl_m_s"),ct=num(mat,"ct_m_s");if(rho!=2700||Math.abs(cl-Math.sqrt(110.3e9/2700))>1e-8||Math.abs(ct-Math.sqrt(26.1e9/2700))>1e-8)throw new IllegalArgumentException("Linear constants do not match material card");
        if(!(s.get("receivers") instanceof List))throw new IllegalArgumentException("Receivers");List<?> rx=(List<?>)s.get("receivers");if(rx.size()<1||rx.size()>8)throw new IllegalArgumentException("Receiver count");
        Map<String,Object> expected=new LinkedHashMap<>();
        put(expected,"schema","workai.comsol.research.case.v1");put(expected,"scenario","guided_plate");put(expected,"boundary","free");
        put(expected,"length_mm",g.get("length_mm"));put(expected,"height_mm",g.get("thickness_mm"));put(expected,"rho",rho);put(expected,"cl",cl);put(expected,"ct",ct);
        for(String k:new String[]{"axisymmetric","conductivity","heat_capacity","expansion","thermal","notch_enabled","notch_x_mm","notch_width_mm","notch_depth_mm","notch_bottom"})put(expected,k,0.0);
        put(expected,"outer_radius_mm",1.0);put(expected,"element_order",4.0);put(expected,"element_budget",40000.0);put(expected,"shot_count",5.0);put(expected,"receiver_count",(double)rx.size());
        for(String k:acq.keySet())put(expected,k,acq.get(k));for(String k:disc.keySet())put(expected,k,disc.get(k));
        put(expected,"bandwidth_estimate_mhz",num(tx,"frequency_mhz")*(1+2/num(tx,"cycles")));
        for(int j=0;j<5;j++){
            String a="tx"+j+"_";put(expected,a+"kind",0.0);put(expected,a+"x_mm",0.0);put(expected,a+"radius_mm",num(g,"thickness_mm")/2);put(expected,a+"direction",0.0);put(expected,a+"frequency_count",1.0);
            put(expected,a+"f0",tx.get("frequency_mhz"));put(expected,a+"amplitude_pa",num(tx,"amplitude_pa")*(j>=3?2:1));for(String k:new String[]{"delay_us","phase_deg","cycles"})put(expected,a+k,tx.get(k));
        }
        Set<String> ids=new HashSet<>();double maxX=0;for(int j=0;j<rx.size();j++){Map<String,Object> r=obj(rx.get(j));keys(r,"id","x_mm");String id=(String)r.get("id");if(!id.matches("[a-zA-Z][a-zA-Z0-9_-]{0,15}")||!ids.add(id))throw new IllegalArgumentException("Receiver ID");maxX=Math.max(maxX,num(r,"x_mm"));put(expected,"rx"+j+"_x_mm",r.get("x_mm"));}
        Map<String,Object> actual=obj(parseJson(pj));if(!actual.keySet().equals(expected.keySet())||p.size()!=expected.size()+4)throw new IllegalArgumentException("Transport fields differ from scene");
        for(String k:expected.keySet()){
            Object v=expected.get(k);if(v instanceof Double){double x=(Double)v;if(!(actual.get(k) instanceof Double)||Math.abs((Double)actual.get(k)-x)>Math.max(1,Math.abs(x))*1e-12||Math.abs(n(k)-x)>Math.max(1,Math.abs(x))*1e-12)throw new IllegalArgumentException("Numeric scene/transport mismatch: "+k);}else{eq(v,actual.get(k));eq(v,p.getProperty(k));}
        }
        range("length_mm",.1,400);range("height_mm",.01,5);range("tx0_f0",.001,2);range("tx0_cycles",4,100);range("tx0_amplitude_pa",1e5,5e6);
        if(n("tx0_f0")*n("sample_step_us")>.1000000001||n("tx0_f0")*n("max_step_us")>.0250000001)throw new IllegalArgumentException("Second harmonic sampling");
        if(n("tx0_delay_us")+n("tx0_cycles")/n("tx0_f0")>n("duration_us")+1e-9||n("duration_us")>=(2*n("length_mm")-maxX)*1000/cl)throw new IllegalArgumentException("Pulse/reflection time window");
        if(5*rx.size()*(Math.round(n("duration_us")/n("sample_step_us"))+1)>4000||5*i("field_nx")*i("field_ny")*i("field_frames")>4000)throw new IllegalArgumentException("Bounded artifact rows");
    }
    private static String nonlinearWeak(Model m){
        m.param().set("lam","58.1e9");m.param().set("mu","26.1e9");m.param().set("nlA","-345e9");m.param().set("nlB","-124.5e9");m.param().set("nlC","-18.5e9");m.param().set("geom_on","0");m.param().set("mat_on","0");
        m.variable().create("nlvars");m.variable("nlvars").model("comp1");
        String[][] F={{"1+ux","uy","0"},{"vx","1+vy","0"},{"wx","wy","1"}};
        for(int a=0;a<3;a++)for(int b=0;b<3;b++)m.variable("nlvars").set("nf"+a+b,F[a][b]);
        for(int a=0;a<3;a++)for(int b=0;b<3;b++){
            StringJoiner terms=new StringJoiner("+");for(int k=0;k<3;k++)terms.add("nf"+k+a+"*nf"+k+b);
            m.variable("nlvars").set("ne"+a+b,"0.5*("+terms+"-"+(a==b?"1":"0")+")");
        }
        m.variable("nlvars").set("ntr","ne00+ne11+ne22");m.variable("nlvars").set("ntr2","ne00^2+ne11^2+ne22^2+2*(ne01^2+ne02^2+ne12^2)");
        m.variable("nlvars").set("nj","nf00*nf11-nf01*nf10");
        for(int a=0;a<3;a++)for(int b=0;b<3;b++){
            StringJoiner square=new StringJoiner("+");for(int k=0;k<3;k++)square.add("ne"+a+k+"*ne"+k+b);
            m.variable("nlvars").set("ns"+a+b,(a==b?"lam*ntr+mat_on*(nlB*ntr2+nlC*ntr^2)+":"")+"(2*mu+2*mat_on*nlB*ntr)*ne"+a+b+"+mat_on*nlA*("+square+")");
        }
        String[] vars={"u","v","w"},dirs={"x","y"};StringJoiner total=new StringJoiner("+","rho*(utt*test(u)+vtt*test(v)+wtt*test(w))+","");
        for(int a=0;a<3;a++)for(int b=0;b<2;b++){
            StringJoiner sum=new StringJoiner("+");for(int k=0;k<3;k++)sum.add("nf"+a+k+"*ns"+k+b);
            String lin=(a==b?"lam*(ux+vy)+":"")+"mu*("+vars[a]+dirs[b]+(a<2?"+"+vars[b]+dirs[a]:"")+")";
            m.variable("nlvars").set("np"+a+b,"if(geom_on>0.5,"+sum+","+lin+")");total.add("np"+a+b+"*test("+vars[a]+dirs[b]+")");
        }
        return total.toString();
    }

    private static double n(String k){double v=Double.parseDouble(p.getProperty(k));if(!Double.isFinite(v))throw new IllegalArgumentException("Nonfinite "+k);return v;}
    private static int i(String k){double v=n(k);if(v!=Math.rint(v))throw new IllegalArgumentException("Integer "+k);return (int)v;}
    private static String f(double x){return Double.toString(x);}
    private static double mm(String k){return n(k)*1e-3;}
    private static void range(String k,double lo,double hi){double v=n(k);if(v<lo||v>hi)throw new IllegalArgumentException("Outside budget: "+k);}
    private static double bottom(){return i("axisymmetric")==1?mm("outer_radius_mm")-mm("height_mm"):0;}
    private static double top(){return bottom()+mm("height_mm");}
    private static void validate(){
        if(!"workai.comsol.research.case.v1".equals(p.getProperty("schema")))throw new IllegalArgumentException("Schema");
        if(!p.getProperty("job_id","").matches("comsol-[a-f0-9]{32}"))throw new IllegalArgumentException("Job ID");
        for(String k:new String[]{"scene_sha256","source_sha256","parameters_sha256"})if(!p.getProperty(k,"").matches("[a-f0-9]{64}"))throw new IllegalArgumentException("Identity");
        if(!Arrays.asList("guided_plate","guided_pipe","weld_array","laser_surface").contains(p.getProperty("scenario")))throw new IllegalArgumentException("Scenario");
        if(!Arrays.asList("free","side_impedance").contains(p.getProperty("boundary")))throw new IllegalArgumentException("Boundary");
        range("length_mm",.1,2000);range("height_mm",.01,200);range("outer_radius_mm",.01,1000);
        range("axisymmetric",0,1);range("thermal",0,1);range("notch_enabled",0,1);range("notch_bottom",0,1);
        if(i("axisymmetric")==1&&bottom()<=0)throw new IllegalArgumentException("Pipe wall reaches axis");
        range("rho",500,20000);range("cl",500,15000);range("ct",100,10000);
        if(n("cl")*n("cl")<=4*n("ct")*n("ct")/3)throw new IllegalArgumentException("Bulk modulus");
        range("shot_count",1,64);range("receiver_count",1,32);range("element_order",4,4);range("element_budget",1,40000);
        range("coarse_h_mm",.0001,100);range("fine_h_mm",.0001,n("coarse_h_mm")*.8);
        range("max_step_us",.000001,10);range("sample_step_us",n("max_step_us"),10);range("duration_us",.01,10000);
        range("relative_tolerance",1e-8,1e-6);range("field_nx",5,101);range("field_ny",3,41);range("field_frames",2,101);
        if(n("duration_us")/(n("max_step_us")*.5)>100000.1)throw new IllegalArgumentException("Step budget");
        if(n("shot_count")*n("receiver_count")*(n("duration_us")/n("sample_step_us")+1)>2000000)throw new IllegalArgumentException("Export budget");
        if((long)i("shot_count")*i("field_nx")*i("field_ny")*i("field_frames")>2000000)throw new IllegalArgumentException("Field export budget");
        if(Math.abs(n("duration_us")/n("sample_step_us")-Math.rint(n("duration_us")/n("sample_step_us")))>1e-6)throw new IllegalArgumentException("Output time grid");
        int intervals=(int)Math.round(n("duration_us")/n("sample_step_us"));
        if(i("field_frames")>intervals+1||intervals%(i("field_frames")-1)!=0)throw new IllegalArgumentException("Uniform field frame grid required");
        if(i("notch_enabled")==1){
            range("notch_depth_mm",.0001,n("height_mm")*.9999);range("notch_width_mm",.0001,n("length_mm")*.9);
            range("notch_x_mm",n("notch_width_mm")/2,n("length_mm")-n("notch_width_mm")/2);
        }
        for(int k=0;k<i("receiver_count");k++)range("rx"+k+"_x_mm",0,n("length_mm"));
        for(int k=0;k<i("shot_count");k++){
            String a="tx"+k+"_";range(a+"kind",0,2);range(a+"x_mm",0,n("length_mm"));range(a+"radius_mm",.0001,n("length_mm")/2);
            if(i(a+"kind")==2){
                if(i("thermal")!=1)throw new IllegalArgumentException("Heat without thermal field");
                range(a+"pulse_us",.00001,n("duration_us"));range(a+"absorbed_j_per_m",1e-12,1000);range(a+"gaussian_integral_m",1e-9,10);
            }else{
                range(a+"frequency_count",1,8);range(a+"cycles",1,100);range(a+"amplitude_pa",1e-6,1e9);
                range(a+"delay_us",0,n("duration_us"));range(a+"phase_deg",-36000,36000);range(a+"direction",0,2);
                for(int j=0;j<i(a+"frequency_count");j++)range(a+"f"+j,.001,100);
                if(i(a+"kind")==1){range(a+"elements",1,128);range(a+"pitch_mm",.0001,n("length_mm"));range(a+"angle_deg",0,89.9);}
            }
        }
        if(i("thermal")==1){range("conductivity",.001,5000);range("heat_capacity",1,10000);range("expansion",1e-10,.001);}
    }
    private static void weak(Model m,String tag,int dim,String expr){
        m.weak().create(tag);m.weak(tag).selection().geom("geom1",dim).all();m.weak(tag).intRule("quad1");
        m.weak(tag).weak((i("axisymmetric")==1?"2*pi*y*(":"(")+expr+")");
    }
    private static String pulse(String a,double delay){
        String t="(t-"+f(delay)+")";StringJoiner sum=new StringJoiner("+","(",")");
        for(int j=0;j<i(a+"frequency_count");j++){
            double freq=n(a+"f"+j)*1e6, duration=n(a+"cycles")/freq;
            sum.add("if(("+t+">=0)&&("+t+"<="+f(duration)+"),0.5*(1-cos(2*pi*"+t+"/"+f(duration)+"))*sin(2*pi*"+f(freq)+"*"+t+"+"+f(n(a+"phase_deg")*Math.PI/180)+"),0)");
        }return sum.toString();
    }
    private static String gaussian(String a,double x){return "exp(-(x-"+f(x)+")^2/"+f(2*Math.pow(mm(a+"radius_mm"),2))+")";}
    private static Model build(String name,double h){
        Model m=ModelUtil.create("Research"+name);m.label("workAI / "+p.getProperty("scenario")+" / "+name);
        for(String k:new String[]{"rho","cl","ct"})m.param().set(k,f(n(k)));
        m.param().set("lam",f(n("rho")*(Math.pow(n("cl"),2)-2*Math.pow(n("ct"),2))));m.param().set("mu",f(n("rho")*Math.pow(n("ct"),2)));
        m.param().set("L",f(mm("length_mm")));m.param().set("Ytop",f(top()));m.param().set("active_tx","0");
        m.component().create("comp1");m.component("comp1").geom().create("geom1",2);
        m.component("comp1").geom("geom1").create("body","Rectangle");
        m.component("comp1").geom("geom1").feature("body").set("size",new double[]{mm("length_mm"),mm("height_mm")});
        m.component("comp1").geom("geom1").feature("body").set("pos",new double[]{0,bottom()});
        if(i("notch_enabled")==1){
            m.component("comp1").geom("geom1").create("notch","Rectangle");
            m.component("comp1").geom("geom1").feature("notch").set("size",new double[]{mm("notch_width_mm"),mm("notch_depth_mm")+1e-8});
            m.component("comp1").geom("geom1").feature("notch").set("pos",new double[]{mm("notch_x_mm")-mm("notch_width_mm")/2,i("notch_bottom")==1?bottom()-1e-8:top()-mm("notch_depth_mm")});
            m.component("comp1").geom("geom1").create("cut","Difference");m.component("comp1").geom("geom1").feature("cut").selection("input").set(new String[]{"body"});m.component("comp1").geom("geom1").feature("cut").selection("input2").set(new String[]{"notch"});
        }
        m.component("comp1").geom("geom1").run();m.component("comp1").mesh().create("mesh1","geom1");
        m.component("comp1").mesh("mesh1").feature("size").set("custom","on");m.component("comp1").mesh("mesh1").feature("size").set("hmax",f(h));
        m.component("comp1").mesh("mesh1").feature("size").set("hmin",f(i("notch_enabled")==1?Math.min(h/4,mm("notch_width_mm")/3):h/4));
        m.component("comp1").mesh("mesh1").feature("size").set("hgrad",1.3);m.component("comp1").mesh("mesh1").create("tri","FreeTri");m.component("comp1").mesh("mesh1").run();
        int ne=m.component("comp1").mesh("mesh1").getNumElem();if(ne>i("element_budget"))throw new IllegalArgumentException("Actual mesh exceeds element budget; no solve started");
        receipt.setProperty(name+"_elements",Integer.toString(ne));
        for(String var:i("thermal")==1?new String[]{"u","v","w","temp"}:new String[]{"u","v","w"}){
            String tag="shape_"+var;m.shape().create(tag,"material1");m.shape(tag).create("lag","shlag");m.shape(tag).feature("lag").set("basename",var).set("order",4);m.shape(tag).selection().geom("geom1",2).all();m.field().create("field_"+var,var);m.field("field_"+var).shape(new String[]{tag});
        }
        m.intRule().create("quad1","material1");m.intRule("quad1").create("order8").order(nonlinear?20:8);
        boolean axi=i("axisymmetric")==1;
        String div=axi?"(ux+vy+v/y)":"(ux+vy)",tdiv=axi?"(test(ux)+test(vy)+test(v)/y)":"(test(ux)+test(vy))";
        String bulk="rho*(utt*test(u)+vtt*test(v)+wtt*test(w))+lam*"+div+"*"+tdiv
            +"+2*mu*(ux*test(ux)+vy*test(vy)"+(axi?"+v/y*test(v)/y":"")+")"
            +"+mu*((uy+vx)*(test(uy)+test(vx))+wx*test(wx)+"+(axi?"(wy-w/y)*(test(wy)-test(w)/y)":"wy*test(wy)")+")";
        if(i("thermal")==1){
            bulk+="-(3*lam+2*mu)*"+f(n("expansion"))+"*temp*"+tdiv;
            weak(m,"heat",2,"rho*"+f(n("heat_capacity"))+"*tempt*test(temp)+"+f(n("conductivity"))+"*(tempx*test(tempx)+tempy*test(tempy))");
        }
        weak(m,"elasticity",2,nonlinear?nonlinearWeak(m):bulk);
        StringJoiner loads=new StringJoiner("+","(",")"),heat=new StringJoiner("+","(",")");boolean hasHeat=false,hasForce=false;
        for(int k=0;k<i("shot_count");k++){
            String a="tx"+k+"_",active="if(active_tx=="+k+",";
            if(i(a+"kind")==2){
                double tau=n(a+"pulse_us")*1e-6;heat.add(active+f(n(a+"absorbed_j_per_m")/n(a+"gaussian_integral_m")/tau)+"*"+gaussian(a,mm(a+"x_mm"))+"*if((t>=0)&&(t<="+f(tau)+"),1-cos(2*pi*t/"+f(tau)+"),0),0)");hasHeat=true;
            }else{
                StringJoiner elements=new StringJoiner("+","(",")");int count=i(a+"kind")==1?i(a+"elements"):1;
                double pitch=count>1?mm(a+"pitch_mm"):0,slow=count>1?Math.sin(Math.toRadians(n(a+"angle_deg")))/n("ct"):0;
                for(int e=0;e<count;e++){
                    double x=mm(a+"x_mm")+(e-(count-1.)/2)*pitch;
                    // Later excitation at increasing x gives kx/omega=sin(theta)/cT.
                    double delay=n(a+"delay_us")*1e-6+e*pitch*slow;
                    elements.add(gaussian(a,x)+"*"+pulse(a,delay));
                }
                String var=new String[]{"u","v","w"}[i(a+"direction")];loads.add(active+f(n(a+"amplitude_pa"))+"*"+elements+",0)*test("+var+")");hasForce=true;
            }
        }
        if(nonlinear){StringJoiner edge=new StringJoiner("+");for(int k=0;k<5;k++){String a="tx"+k+"_";edge.add("if(active_tx=="+k+","+f(n(a+"amplitude_pa"))+"*"+pulse(a,n(a+"delay_us")*1e-6)+",0)");}weak(m,"traction",1,"-if(abs(x)<1e-9,("+edge+")*test(u),0)");}
        else if(hasForce)weak(m,"traction",1,"-if(abs(y-Ytop)<1e-9,"+loads+",0)");
        if(hasHeat)weak(m,"heat_flux",1,"-if(abs(y-Ytop)<1e-9,"+heat+"*test(temp),0)");
        if("side_impedance".equals(p.getProperty("boundary")))weak(m,"sides",1,"if((abs(x)<1e-9)||(abs(x-L)<1e-9),rho*(cl*ut*test(u)+ct*(vt*test(v)+wt*test(w))),0)");
        return m;
    }
    private static double[][][] evaluate(Model m,String tag,double[][] coords,int[] solutions){
        if(Arrays.asList(m.result().numerical().tags()).contains(tag))m.result().numerical().remove(tag);
        m.result().numerical().create(tag,"Interp");m.result().numerical(tag).set("expr",new String[]{"t","u","v","w",i("thermal")==1?"temp":"0"});
        m.result().numerical(tag).set("data","dset1");m.result().numerical(tag).set("edim",2);m.result().numerical(tag).selection().all();m.result().numerical(tag).set("coord",coords);m.result().numerical(tag).set("solnum",solutions);return m.result().numerical(tag).getData();
    }
    private static double at(double[][][] v,int e,int t,int p,int nt,int np){
        double x;if(v.length!=5)throw new IllegalStateException("Expected 5 expressions");
        if(v[0].length==nt&&v[0][0].length==np)x=v[e][t][p];else if(v[0].length==np&&v[0][0].length==nt)x=v[e][p][t];else throw new IllegalStateException("Interpolation shape");
        if(!Double.isFinite(x))throw new IllegalStateException("Nonfinite result");return x;
    }
    private static boolean cut(double x,double y){return i("notch_enabled")==1&&Math.abs(x-mm("notch_x_mm"))<=mm("notch_width_mm")/2+1e-10&&(i("notch_bottom")==1?y<=bottom()+mm("notch_depth_mm")+1e-10:y>=top()-mm("notch_depth_mm")-1e-10);}
    private static void saveCleared(Model m,String name)throws Exception{
        String[] sol=m.sol().tags(),studies=m.study().tags();Map<String,String[]> feat=new LinkedHashMap<>();
        if(sol.length==0)throw new IllegalStateException("No solver");
        for(String s:sol){feat.put(s,m.sol(s).feature().tags());m.sol(s).clearSolutionData();if(!m.sol(s).isEmpty())throw new IllegalStateException("Clear failed");}
        m.save(name+".mph");Model check=ModelUtil.loadCopy("Check"+name,name+".mph");
        try{if(!Arrays.equals(sol,check.sol().tags())||!Arrays.equals(studies,check.study().tags()))throw new IllegalStateException("Reopen configuration differs");for(String s:sol)if(!check.sol(s).isEmpty()||!Arrays.equals(feat.get(s),check.sol(s).feature().tags()))throw new IllegalStateException("Reopen solution not empty");}finally{ModelUtil.remove(check.tag());}
    }
    private static void solve(String level,double h,double step)throws Exception{
        long start=System.nanoTime();Model m=build(level,h);receipt.setProperty(level+"_setup_seconds",f((System.nanoTime()-start)*1e-9));
        int nt=(int)Math.round(n("duration_us")/n("sample_step_us"))+1,nr=i("receiver_count"),nx=i("field_nx"),ny=i("field_ny"),nf=i("field_frames");
        double dt=n("sample_step_us")*1e-6;int[] all=new int[nt],frames=new int[nf];for(int j=0;j<nt;j++)all[j]=j+1;for(int j=0;j<nf;j++)frames[j]=(int)Math.round((nt-1.)*j/(nf-1))+1;
        String times="range(0,"+f(dt)+","+f(n("duration_us")*1e-6)+")";
        m.study().create("std1");m.study("std1").create("time","Transient");m.study("std1").feature("time").set("tlist",times);m.study("std1").createAutoSequences("all");
        String solver=m.sol().tags()[0],tag=null;for(String s:m.sol(solver).feature().tags())if("Time".equals(m.sol(solver).feature(s).getType()))tag=s;if(tag==null)throw new IllegalStateException("Time solver missing");
        m.sol(solver).feature(tag).set("tlist",times);m.sol(solver).feature(tag).set("timemethod","genalpha");m.sol(solver).feature(tag).set("tstepsgenalpha","manual");m.sol(solver).feature(tag).set("timestepgenalpha",f(step));m.sol(solver).feature(tag).set("rhoinf",1.0);
        m.sol(solver).feature(tag).set("rtol",n("relative_tolerance")*(level.equals("timecheck")?.25:1));m.sol(solver).feature(tag).set("atolglobalvaluemethod","manual");m.sol(solver).feature(tag).set("atolglobalmethod","unscaled");m.sol(solver).feature(tag).set("atolglobal",nonlinear?(level.equals("timecheck")?5e-17:2e-16):(level.equals("timecheck")?2.5e-15:1e-14));
        receipt.setProperty(level+"_internal_step_s",f(step));receipt.setProperty(level+"_mesh_h_m",f(h));
        double[][] rx=new double[2][nr];for(int r=0;r<nr;r++){rx[0][r]=Math.max(1e-9,Math.min(mm("length_mm")-1e-9,mm("rx"+r+"_x_mm")));rx[1][r]=top()-1e-9;if(cut(rx[0][r],rx[1][r]))throw new IllegalArgumentException("Receiver in void");}
        List<double[]> points=new ArrayList<>();List<int[]> indices=new ArrayList<>();
        for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
            double px=mm("length_mm")*x/(nx-1),py=bottom()+mm("height_mm")*y/(ny-1);if(cut(px,py))continue;
            points.add(new double[]{Math.max(1e-9,Math.min(mm("length_mm")-1e-9,px)),Math.max(bottom()+1e-9,Math.min(top()-1e-9,py))});indices.add(new int[]{x,y});
        }
        double[][] coords=new double[2][points.size()];for(int j=0;j<points.size();j++){coords[0][j]=points.get(j)[0];coords[1][j]=points.get(j)[1];}
        double solveSeconds=0;
        try(PrintWriter out=new PrintWriter(Files.newBufferedWriter(Paths.get(level+".csv")));PrintWriter field=new PrintWriter(Files.newBufferedWriter(Paths.get(level+".field.csv")))){
            out.println("shot,rx,time_s,u_m,v_m,w_m,temperature_rise_k");field.println("shot,frame,ix,iy,time_s,u_m,v_m,w_m,temperature_rise_k");
            for(int shot=0;shot<i("shot_count");shot++){
                m.param().set("active_tx",Integer.toString(shot));if(nonlinear){m.param().set("geom_on",shot==0?"0":"1");m.param().set("mat_on",shot==2||shot==4?"1":"0");}System.out.println("WORKAI_RESEARCH_SOLVE "+level+" shot="+shot);long tick=System.nanoTime();m.sol(solver).runAll();solveSeconds+=(System.nanoTime()-tick)*1e-9;
                double[][][] values=evaluate(m,"receivers",rx,all);
                for(int t=0;t<nt;t++)for(int r=0;r<nr;r++){
                    double time=at(values,0,t,r,nt,nr);if(Math.abs(time-t*dt)>1e-11)throw new IllegalStateException("Time grid");
                    out.printf(Locale.ROOT,"%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g%n",shot,r,time,at(values,1,t,r,nt,nr),at(values,2,t,r,nt,nr),at(values,3,t,r,nt,nr),at(values,4,t,r,nt,nr));
                }
                values=evaluate(m,"wavefield",coords,frames);
                for(int t=0;t<nf;t++)for(int r=0;r<points.size();r++){
                    double time=at(values,0,t,r,nf,points.size());if(Math.abs(time-(frames[t]-1)*dt)>1e-11)throw new IllegalStateException("Frame grid");int[] xy=indices.get(r);
                    field.printf(Locale.ROOT,"%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g%n",shot,t,xy[0],xy[1],time,at(values,1,t,r,nf,points.size()),at(values,2,t,r,nf,points.size()),at(values,3,t,r,nf,points.size()),at(values,4,t,r,nf,points.size()));
                }
                if(nonlinear){
                    if(!Arrays.asList(m.result().numerical().tags()).contains("strainGuard"))m.result().numerical().create("strainGuard","MaxSurface");
                    m.result().numerical("strainGuard").set("expr",new String[]{"sqrt(ntr2)","-nj"});m.result().numerical("strainGuard").set("data","dset1");m.result().numerical("strainGuard").selection().all();m.result().numerical("strainGuard").set("solnum",all);
                    double[][] bounds=m.result().numerical("strainGuard").getReal();if(bounds.length!=2)throw new IllegalStateException("Strain guard shape");double strain=0,minJ=Double.POSITIVE_INFINITY;
                    for(double x:bounds[0]){if(!Double.isFinite(x))throw new IllegalStateException("Nonfinite strain");strain=Math.max(strain,x);}for(double x:bounds[1]){if(!Double.isFinite(x))throw new IllegalStateException("Nonfinite Jacobian");minJ=Math.min(minJ,-x);}
                    if(strain>0.001||minJ<=0)throw new IllegalStateException("Outside small elastic strain validation budget");
                    receipt.setProperty(level+"_shot"+shot+"_max_green_strain",f(strain));receipt.setProperty(level+"_shot"+shot+"_min_J",f(minJ));
                }
                out.flush();field.flush();m.sol(solver).clearSolutionData();
            }if(out.checkError()||field.checkError())throw new IOException("CSV write failure");
        }
        receipt.setProperty(level+"_solve_seconds",f(solveSeconds));saveCleared(m,level);ModelUtil.remove(m.tag());System.out.println("WORKAI_RESEARCH_CLEARED_AND_REOPENED "+level);
    }
    public static Model run()throws Exception{
        p=new Properties();try(InputStream in=Files.newInputStream(Paths.get("case.properties"))){p.load(in);}validate();readScene();if(Files.exists(Paths.get("completed.properties")))throw new IOException("Cannot rerun completed job");
        receipt=new Properties();for(String k:new String[]{"job_id","scene_sha256","source_sha256","parameters_sha256","scenario","axisymmetric","thermal"})receipt.setProperty(k,p.getProperty(k));
        solve("coarse",mm("coarse_h_mm"),n("max_step_us")*1e-6);solve("fine",mm("fine_h_mm"),n("max_step_us")*1e-6);solve("timecheck",mm("fine_h_mm"),n("max_step_us")*.5e-6);
        receipt.setProperty("schema","workai.comsol.research.receipt.v1");receipt.setProperty("method",nonlinear?"comsol_murnaghan_total_lagrangian_2d_v1":"comsol_research_2d_axisymmetric_weakform_v1");receipt.setProperty("status","completed");receipt.setProperty("comsol_version",ModelUtil.getComsolVersion());receipt.setProperty("completed_at_utc",java.time.Instant.now().toString());receipt.setProperty("mph_solution_state","cleared");receipt.setProperty("mph_reopen_verified","true");receipt.setProperty("element_order","4");receipt.setProperty("quadrature_order",nonlinear?"20":"8");receipt.setProperty("time_integrator","generalized_alpha_manual");receipt.setProperty("rho_infinity","1.0");receipt.setProperty("quantity","displacement_m_temperature_rise_k");
        if(nonlinear){receipt.setProperty("constitutive","murnaghan_green_lagrange");receipt.setProperty("energy_definition","murnaghan_invariants_reference_volume");receipt.setProperty("source_profile","left_edge_uniform_nominal_traction");receipt.setProperty("controls",String.join(",",CONTROLS));receipt.setProperty("material_record_sha256","39102235cd2fae341a277e46aa08573ae2294710c79abd104b10491fe5edef0a");receipt.setProperty("lambda_pa","5.81e10");receipt.setProperty("mu_pa","2.61e10");receipt.setProperty("l_pa","-1.43e11");receipt.setProperty("m_pa","-2.97e11");receipt.setProperty("n_pa","-3.45e11");receipt.setProperty("spatial_assumption","plane_strain_z_invariant");}
        try(OutputStream out=Files.newOutputStream(Paths.get("completed.properties"))){receipt.store(out,"Real COMSOL solves, export and cleared MPH reopen required");}return null;
    }
    public static void main(String[] args)throws Exception{run();}
}
