// COMSOL 6.4 research model: Cartesian 2D P-SV/SH, cylindrical axisymmetric
// elasticity and one-way thermoelasticity. Numeric input only; no supplied expressions.
import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class UltraResearch {
    private static Properties p, receipt;
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
        m.intRule().create("quad1","material1");m.intRule("quad1").create("order8").order(8);
        boolean axi=i("axisymmetric")==1;
        String div=axi?"(ux+vy+v/y)":"(ux+vy)",tdiv=axi?"(test(ux)+test(vy)+test(v)/y)":"(test(ux)+test(vy))";
        String bulk="rho*(utt*test(u)+vtt*test(v)+wtt*test(w))+lam*"+div+"*"+tdiv
            +"+2*mu*(ux*test(ux)+vy*test(vy)"+(axi?"+v/y*test(v)/y":"")+")"
            +"+mu*((uy+vx)*(test(uy)+test(vx))+wx*test(wx)+"+(axi?"(wy-w/y)*(test(wy)-test(w)/y)":"wy*test(wy)")+")";
        if(i("thermal")==1){
            bulk+="-(3*lam+2*mu)*"+f(n("expansion"))+"*temp*"+tdiv;
            weak(m,"heat",2,"rho*"+f(n("heat_capacity"))+"*tempt*test(temp)+"+f(n("conductivity"))+"*(tempx*test(tempx)+tempy*test(tempy))");
        }
        weak(m,"elasticity",2,bulk);
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
        if(hasForce)weak(m,"traction",1,"-if(abs(y-Ytop)<1e-9,"+loads+",0)");
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
        m.sol(solver).feature(tag).set("rtol",n("relative_tolerance")*(level.equals("timecheck")?.25:1));m.sol(solver).feature(tag).set("atolglobalvaluemethod","manual");m.sol(solver).feature(tag).set("atolglobalmethod","unscaled");m.sol(solver).feature(tag).set("atolglobal",level.equals("timecheck")?2.5e-15:1e-14);
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
                m.param().set("active_tx",Integer.toString(shot));System.out.println("WORKAI_RESEARCH_SOLVE "+level+" shot="+shot);long tick=System.nanoTime();m.sol(solver).runAll();solveSeconds+=(System.nanoTime()-tick)*1e-9;
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
                out.flush();field.flush();m.sol(solver).clearSolutionData();
            }if(out.checkError()||field.checkError())throw new IOException("CSV write failure");
        }
        receipt.setProperty(level+"_solve_seconds",f(solveSeconds));saveCleared(m,level);ModelUtil.remove(m.tag());System.out.println("WORKAI_RESEARCH_CLEARED_AND_REOPENED "+level);
    }
    public static Model run()throws Exception{
        p=new Properties();try(InputStream in=Files.newInputStream(Paths.get("case.properties"))){p.load(in);}validate();if(Files.exists(Paths.get("completed.properties")))throw new IOException("Cannot rerun completed job");
        receipt=new Properties();for(String k:new String[]{"job_id","scene_sha256","source_sha256","parameters_sha256","scenario","axisymmetric","thermal"})receipt.setProperty(k,p.getProperty(k));
        solve("coarse",mm("coarse_h_mm"),n("max_step_us")*1e-6);solve("fine",mm("fine_h_mm"),n("max_step_us")*1e-6);solve("timecheck",mm("fine_h_mm"),n("max_step_us")*.5e-6);
        receipt.setProperty("schema","workai.comsol.research.receipt.v1");receipt.setProperty("method","comsol_research_2d_axisymmetric_weakform_v1");receipt.setProperty("status","completed");receipt.setProperty("comsol_version",ModelUtil.getComsolVersion());receipt.setProperty("completed_at_utc",java.time.Instant.now().toString());receipt.setProperty("mph_solution_state","cleared");receipt.setProperty("mph_reopen_verified","true");receipt.setProperty("element_order","4");receipt.setProperty("quadrature_order","8");receipt.setProperty("time_integrator","generalized_alpha_manual");receipt.setProperty("rho_infinity","1.0");receipt.setProperty("quantity","displacement_m_temperature_rise_k");
        try(OutputStream out=Files.newOutputStream(Paths.get("completed.properties"))){receipt.store(out,"Real COMSOL solves, export and cleared MPH reopen required");}return null;
    }
    public static void main(String[] args)throws Exception{run();}
}
