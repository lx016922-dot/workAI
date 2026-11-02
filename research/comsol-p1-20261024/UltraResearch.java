// COMSOL reference for the uploaded 2026-10-24 P1 request. SI units.
// Total Lagrangian weak form; all loads and receiver positions use reference coordinates.
import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
public class UltraResearch {
 private static Properties p,receipt;
 private static Map<String,Object> scene,cfg,material;
 private static List<?> receivers,laws;
 private static boolean selfcheck;
 private static String family,level;
 private static double L,W,H,h,step,duration,sample;
 private static int quadrature=20;
 private static int[] loadFaces,symFaces;
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
 private static String f(double x){if(!Double.isFinite(x))throw new IllegalArgumentException("Nonfinite value");return Double.toString(x);}
 private static double n(String k){return Double.parseDouble(p.getProperty(k));}
 private static double val(Map<String,Object> m,String k,double lo,double hi){double x=num(m,k);if(x<lo||x>hi)throw new IllegalArgumentException("Outside 3D budget: "+k);return x;}
 private static double[] vector(Object x){List<?> a=(List<?>)x;if(a.size()!=3)throw new IllegalArgumentException("3-vector required");return new double[]{(Double)a.get(0),(Double)a.get(1),(Double)a.get(2)};}
 private static void close(double a,double b,String name){if(Math.abs(a-b)>1e-12*Math.max(1,Math.abs(b)))throw new IllegalArgumentException("Transport/scene mismatch: "+name);}

 private static void read()throws Exception{
  p=new Properties();try(InputStream in=Files.newInputStream(Paths.get("case.properties"))){p.load(in);}
  Map<String,Object> request=obj(parseJson(new String(Files.readAllBytes(Paths.get("request.json")),java.nio.charset.StandardCharsets.UTF_8)));
  for(String k:new String[]{"job_id","source_sha256","scene_sha256","parameters_sha256"})eq(request.get(k),p.getProperty(k));
  String sj=(String)request.get("scene_json"),pj=(String)request.get("parameters_json");eq(sha(sj),p.getProperty("scene_sha256"));eq(sha(pj),p.getProperty("parameters_sha256"));
  Map<String,Object> root=obj(parseJson(sj));keys(root,"schema","case_id","scene","execution");eq(root.get("schema"),"workai.comsol.graphwave3d.scene.v1");
  scene=obj(root.get("scene"));cfg=obj(root.get("execution"));eq(scene.get("schema"),"workai.comsol.p1.v1");
  selfcheck=scene.get("analysis").equals("material_check");if(!selfcheck)eq(scene.get("analysis"),"transient");
  family=(String)scene.get("family");if(!Arrays.asList("A","B","C").contains(family))throw new IllegalArgumentException("Family");
  Map<String,Object> g=obj(scene.get("geometry"));L=num(g,"length_mm")*1e-3;W=num(g,"width_mm")*1e-3;H=num(g,"thickness_mm")*1e-3;
  close(L,selfcheck?.001:.048,"L");close(W,selfcheck?.001:.002,"W");if(selfcheck)close(H,.001,"H");else if(Math.abs(H-.0016)>1e-12&&Math.abs(H-.002)>1e-12)throw new IllegalArgumentException("P1 thickness");
  material=obj(scene.get("material"));String[] mk={"rho_kg_m3","lambda_pa","mu_pa","l_pa","m_pa","n_pa"};double[] mv={2700,58.1e9,26.1e9,-143e9,-297e9,-345e9};for(int j=0;j<mk.length;j++)close(num(material,mk[j]),mv[j],mk[j]);
  close(num(scene,"alpha_max_per_s"),22075615.536638446,"alpha");close(num(scene,"amplitude_pa"),1e6,"amplitude");close(num(scene,"cycles"),5,"cycles");
  close(num(scene,"frequency_mhz"),family.equals("C")?.25:.4,"frequency");
  if(!selfcheck){eq(scene.get("symmetry"),"ymax_uy_zero_ux_uz_free");eq(scene.get("absorbing_faces"),family.equals("B")?"xmax":"xmin+xmax");}
  h=num(cfg,"h_mm")*1e-3;step=num(cfg,"max_step_us")*1e-6;duration=num(cfg,"duration_us")*1e-6;sample=num(cfg,"sample_step_us")*1e-6;
  if(selfcheck)close(h,.001,"selfcheck h");else if(Math.abs(h-.0008)>1e-12&&Math.abs(h-.00056)>1e-12)throw new IllegalArgumentException("Mesh variant");
  if(Math.abs(step-1e-8)>1e-18&&Math.abs(step-5e-9)>1e-18)throw new IllegalArgumentException("Step variant");close(duration,30e-6,"duration");close(sample,.05e-6,"sample");
  close(num(cfg,"quadrature_order"),20,"quadrature");close(num(cfg,"degree"),4,"degree");val(cfg,"relative_tolerance",1e-10,1e-6);val(cfg,"absolute_tolerance_m",1e-20,1e-18);
  level=(String)cfg.get("level");if(!Arrays.asList("coarse","fine","timecheck").contains(level))throw new IllegalArgumentException("Output level");
  laws=(List<?>)cfg.get("laws");Set<Integer> used=new HashSet<>();for(Object law:laws){double d=(Double)law;int k=(int)d;if(k!=d||k<0||k>2||!used.add(k))throw new IllegalArgumentException("Controls");}if(laws.isEmpty())throw new IllegalArgumentException("No controls");
  receivers=(List<?>)scene.get("receivers");if(receivers.size()!=13)throw new IllegalArgumentException("13 receivers required");
  eq(p.getProperty("schema"),"workai.comsol.research.case.v1");eq(p.getProperty("boundary"),"free");close(n("axisymmetric"),0,"axisymmetric");close(n("thermal"),0,"thermal");
  close(n("length_mm"),L*1e3,"length");close(n("height_mm"),H*1e3,"thickness");close(n("rho"),2700,"rho");close(n("cl"),Math.sqrt(110.3e9/2700),"cl");close(n("ct"),Math.sqrt(26.1e9/2700),"ct");
  close(n("duration_us"),duration*1e6,"duration");close(n("sample_step_us"),sample*1e6,"sample");close(n("max_step_us"),step*1e6,"step");close(n("element_order"),4,"degree");close(n("shot_count"),laws.size(),"shots");close(n("receiver_count"),receivers.size(),"receivers");
  for(int j=0;j<13;j++)close(n("rx"+j+"_x_mm"),vector(obj(receivers.get(j)).get("point_mm"))[0],"receiver x");
  if(Files.exists(Paths.get("completed.properties")))throw new IllegalArgumentException("Completed job immutable");
  receipt=new Properties();for(String k:new String[]{"job_id","scene_sha256","source_sha256","parameters_sha256"})receipt.setProperty(k,p.getProperty(k));receipt.setProperty("case_id",(String)root.get("case_id"));
 }
 private static int[] face(Model m,String tag,int axis,double coordinate){
  m.component("comp1").selection().create(tag,"Box");m.component("comp1").selection(tag).set("entitydim",2);String[] dim={"x","y","z"};
  for(int k=0;k<3;k++){m.component("comp1").selection(tag).set(dim[k]+"min",k==axis?coordinate-1e-9:-.1);m.component("comp1").selection(tag).set(dim[k]+"max",k==axis?coordinate+1e-9:.1);}
  m.component("comp1").selection(tag).set("condition","allvertices");int[] ids=m.component("comp1").selection(tag).entities(2);if(ids.length!=1)throw new IllegalStateException("Expected one face "+tag);receipt.setProperty(tag+"_boundaries",Arrays.toString(ids));return ids;
 }
 private static String pulse(double delay){double frequency=num(scene,"frequency_mhz")*1e6,T=5/frequency;String tau="(t-"+f(delay)+")";return "if((t>="+f(delay)+")&&(t<="+f(delay+T)+"),1e6*0.5*(1-cos(2*pi*"+f(frequency/5)+"*"+tau+"))*sin(2*pi*"+f(frequency)+"*"+tau+"),0)";}
 private static String gaussian(double x,double sigma){return "exp(-((x-"+f(x)+")^2+(y-0.001)^2)/"+f(2*sigma*sigma)+")";}
 private static void variable(Model m,String key,String expression){m.variable("vars").set(key,expression);}
 private static Model build(){
  Model m=ModelUtil.create("P1Reference");m.label("P1 half-width plate; uploaded 2026-10-24 contract");m.component().create("comp1");m.component("comp1").geom().create("geom1",3);m.component("comp1").geom("geom1").create("body","Block");m.component("comp1").geom("geom1").feature("body").set("size",new double[]{L,W,H});m.component("comp1").geom("geom1").run();
  m.component("comp1").mesh().create("mesh1","geom1");MeshSequence mesh=m.component("comp1").mesh("mesh1");int[] start=face(m,"meshStart",0,0),end=face(m,"meshEnd",0,L);mesh.create("map","Map");mesh.feature("map").selection().set(start);
  double[] size={L,W,H};int[] counts=new int[3];for(int axis=0;axis<3;axis++){counts[axis]=(int)Math.ceil(size[axis]/h-1e-10);receipt.setProperty("elements_axis_"+axis,Integer.toString(counts[axis]));receipt.setProperty("h_axis_"+axis+"_mm",f(size[axis]/counts[axis]*1e3));}
  for(int axis=1;axis<3;axis++){String tag="edge"+axis;m.component("comp1").selection().create(tag,"Box");m.component("comp1").selection(tag).set("entitydim",1);String[] dim={"x","y","z"};for(int k=0;k<3;k++){m.component("comp1").selection(tag).set(dim[k]+"min",-1e-9);m.component("comp1").selection(tag).set(dim[k]+"max",k==axis?size[k]+1e-9:1e-9);}m.component("comp1").selection(tag).set("condition","allvertices");int[] ids=m.component("comp1").selection(tag).entities(1);if(ids.length!=1)throw new IllegalStateException("Mapped edge");mesh.feature("map").create(tag,"Distribution");mesh.feature("map").feature(tag).selection().set(ids);mesh.feature("map").feature(tag).set("numelem",counts[axis]);mesh.feature("map").feature(tag).set("equidistant","on");}
  mesh.create("sweep","Sweep");mesh.feature("sweep").selection("sourceface").set(start);mesh.feature("sweep").selection("targetface").set(end);mesh.feature("sweep").create("layers","Distribution");mesh.feature("sweep").feature("layers").selection().all();mesh.feature("sweep").feature("layers").set("numelem",counts[0]);mesh.feature("sweep").feature("layers").set("equidistant","on");mesh.run();
  int ne=mesh.getNumElem();if(ne>n("element_budget"))throw new IllegalArgumentException("Element budget");receipt.setProperty("elements",Integer.toString(ne));receipt.setProperty("volume_hexahedra",Integer.toString(counts[0]*counts[1]*counts[2]));
  for(String v:new String[]{"u","v","w"}){m.shape().create("shape_"+v,"material1");m.shape("shape_"+v).create("lag","shlag");m.shape("shape_"+v).feature("lag").set("basename",v).set("order",4);m.shape("shape_"+v).selection().geom("geom1",3).all();m.field().create("field_"+v,v);m.field("field_"+v).shape(new String[]{"shape_"+v});}
  m.intRule().create("quad","material1");m.intRule("quad").create("order").order(quadrature);
  m.param().set("rho",f(num(material,"rho_kg_m3")));m.param().set("lam",f(num(material,"lambda_pa")));m.param().set("mu",f(num(material,"mu_pa")));m.param().set("aa",f(num(material,"n_pa")));m.param().set("bb",f(num(material,"m_pa")-num(material,"n_pa")/2));m.param().set("cc",f(num(material,"l_pa")-num(material,"m_pa")+num(material,"n_pa")/2));m.param().set("geom_on","0");m.param().set("mat_on","0");
  m.variable().create("vars");m.variable("vars").model("comp1");String[] vars={"u","v","w"},dirs={"x","y","z"};
  for(int a=0;a<3;a++)for(int b=0;b<3;b++)variable(m,"gf"+a+b,(a==b?"1+":"")+vars[a]+dirs[b]);
  // Algebraically identical Green strain, without subtracting nearly equal ones.
  for(int a=0;a<3;a++)for(int b=0;b<3;b++){StringJoiner sum=new StringJoiner("+");for(int k=0;k<3;k++)sum.add(vars[k]+dirs[a]+"*"+vars[k]+dirs[b]);variable(m,"gl"+a+b,"0.5*("+vars[a]+dirs[b]+"+"+vars[b]+dirs[a]+")");variable(m,"ge"+a+b,"gl"+a+b+"+0.5*("+sum+")");}
  variable(m,"trE","ge00+ge11+ge22");variable(m,"trE2","ge00^2+ge11^2+ge22^2+2*(ge01^2+ge02^2+ge12^2)");variable(m,"trL","gl00+gl11+gl22");variable(m,"trL2","gl00^2+gl11^2+gl22^2+2*(gl01^2+gl02^2+gl12^2)");
  StringJoiner tr3=new StringJoiner("+");for(int a=0;a<3;a++)for(int b=0;b<3;b++)for(int c=0;c<3;c++)tr3.add("ge"+a+b+"*ge"+b+c+"*ge"+c+a);variable(m,"trE3",tr3.toString());
  variable(m,"detF","gf00*(gf11*gf22-gf12*gf21)-gf01*(gf10*gf22-gf12*gf20)+gf02*(gf10*gf21-gf11*gf20)");
  for(int a=0;a<3;a++)for(int b=0;b<3;b++){StringJoiner sq=new StringJoiner("+");for(int k=0;k<3;k++)sq.add("ge"+a+k+"*ge"+k+b);variable(m,"gs"+a+b,(a==b?"lam*trE+mat_on*(bb*trE2+cc*trE^2)+":"")+"(2*mu+2*mat_on*bb*trE)*ge"+a+b+"+mat_on*aa*("+sq+")");}
  StringJoiner bulk=new StringJoiner("+",selfcheck?"":"rho*(utt*test(u)+vtt*test(v)+wtt*test(w)+alpha*(ut*test(u)+vt*test(v)+wt*test(w)))+","");
  for(int a=0;a<3;a++)for(int b=0;b<3;b++){StringJoiner sum=new StringJoiner("+");for(int k=0;k<3;k++)sum.add("gf"+a+k+"*gs"+k+b);variable(m,"gp"+a+b,"if(geom_on>0.5,"+sum+","+(a==b?"lam*trL+":"")+"2*mu*gl"+a+b+")");bulk.add("gp"+a+b+"*test("+vars[a]+dirs[b]+")");}
  variable(m,"energyW","if(geom_on>0.5,0.5*lam*trE^2+mu*trE2+mat_on*(aa/3*trE3+bb*trE*trE2+cc/3*trE^3),0.5*lam*trL^2+mu*trL2)");variable(m,"energyK","0.5*rho*(ut^2+vt^2+wt^2)");
  m.weak().create("bulk");m.weak("bulk").selection().geom("geom1",3).all();m.weak("bulk").intRule("quad");m.weak("bulk").weak(bulk.toString());

  variable(m,"alpha",selfcheck?"0":f(num(scene,"alpha_max_per_s"))+"*("+(family.equals("B")?"0":"if(x<0.008,((0.008-x)/0.008)^3,0)")+"+if(x>0.040,((x-0.040)/0.008)^3,0))");
  if(selfcheck){
   m.param().set("geom_on","1");m.param().set("mat_on","1");String[] expressions=new String[3];String[] xyz={"x","y","z"};
   for(int a=0;a<3;a++){StringJoiner expression=new StringJoiner("+");for(int b=0;b<3;b++){m.param().set("d"+a+b,"0");expression.add("d"+a+b+"*"+xyz[b]);}expressions[a]=vars[a]+"-("+expression+")";}
   m.constr().create("uniform",new String[]{"shape_u","shape_v","shape_w"});ConstrFeature c=m.constr("uniform").create("affine");c.set("constr",expressions);c.selection().geom("geom1",2).all();
  }else{
   loadFaces=face(m,"load",family.equals("B")?0:2,family.equals("B")?0:H);
   String drive=family.equals("A")?"-"+gaussian(.016,.002)+"*"+pulse(0):family.equals("B")?pulse(0):"-"+gaussian(.013,.0015)+"*"+pulse(0)+"+"+gaussian(.019,.0015)+"*"+pulse(1e-6);
   variable(m,"drive",drive);m.weak().create("load");m.weak("load").selection().geom("geom1",2).set(loadFaces);m.weak("load").intRule("quad");m.weak("load").weak("-drive*test("+(family.equals("B")?"u":"w")+")");
   symFaces=face(m,"symmetry",1,W);m.constr().create("symmetry",new String[]{"shape_v"});ConstrFeature c=m.constr("symmetry").create("zero");c.set("constr",new String[]{"v"});c.selection().geom("geom1",2).set(symFaces);
  }return m;
 }
 private static void describe(SolverFeature feature,String prefix){
  System.out.println("P1_PROPERTIES "+prefix+" "+Arrays.toString(feature.properties()));
  for(String key:feature.properties())if(key.contains("tol")||key.contains("step")||key.equals("rhoinf")||key.equals("maxiter")||key.equals("termonres")||key.equals("linsolver")||key.equals("jtech")){try{System.out.println("P1_PROPERTY "+prefix+" "+key+"="+feature.getString(key)+" allowed="+Arrays.toString(feature.getAllowedPropertyValues(key)));}catch(Exception ignored){}}
  for(String tag:feature.feature().tags())describe(feature.feature(tag),prefix+"/"+tag);
 }
 private static String timeSolver(Model m,boolean configure){
  String times="range(0,"+f(sample)+","+f(duration)+")";m.study().create("stdTime");m.study("stdTime").create("time","Transient");m.study("stdTime").feature("time").set("tlist",times);m.study("stdTime").createAutoSequences("all");String[] solvers=m.sol().tags();String solver=solvers[solvers.length-1],time=null;
  for(String tag:m.sol(solver).feature().tags())if(m.sol(solver).feature(tag).getType().equals("Time"))time=tag;if(time==null)throw new IllegalStateException("Time solver");SolverFeature t=m.sol(solver).feature(time);
  t.set("timemethod","genalpha");t.set("rhoinf",1.0);t.set("rtol",num(cfg,"relative_tolerance"));t.set("atolglobalvaluemethod","manual");t.set("atolglobalmethod","unscaled");t.set("atolglobal",num(cfg,"absolute_tolerance_m"));t.set("reacf","off");
  if(configure){
   t.set("tstepsgenalpha","strict");t.set("initialstepgenalphaactive","on");t.set("initialstepgenalpha",f(step));t.set("maxstepconstraintgenalpha","const");t.set("maxstepgenalpha",f(step));
   for(String tag:t.feature().tags())if(t.feature(tag).getType().equals("FullyCoupled")){t.feature(tag).set("jtech","once");t.feature(tag).set("ntolfact",.001);t.feature(tag).set("maxiter",12);t.feature(tag).set("termonres","off");}
   receipt.setProperty("time_steps_mode","strict");receipt.setProperty("relative_tolerance",f(num(cfg,"relative_tolerance")));receipt.setProperty("absolute_tolerance_m",f(num(cfg,"absolute_tolerance_m")));receipt.setProperty("newton_tolerance_factor","0.001");receipt.setProperty("newton_termination","solution_change");receipt.setProperty("rho_infinity","1.0");
  }
  describe(t,"time");return solver;
 }
 private static double[][] evaluate(Model m,String tag,String type,String[] expressions,int[] solutions){
  if(Arrays.asList(m.result().numerical().tags()).contains(tag))m.result().numerical().remove(tag);m.result().numerical().create(tag,type);m.result().numerical(tag).set("expr",expressions);m.result().numerical(tag).set("data","dset1");m.result().numerical(tag).selection().all();m.result().numerical(tag).set("solnum",solutions);return m.result().numerical(tag).getReal();
 }
 private static void finish(Model m,String solver)throws Exception{
  m.sol(solver).clearSolutionData();m.save(level+".mph");ModelUtil.remove(m.tag());receipt.setProperty("schema","workai.comsol.p1.receipt.v1");receipt.setProperty("status","completed");receipt.setProperty("comsol_version",ModelUtil.getComsolVersion());receipt.setProperty("completed_at_utc",java.time.Instant.now().toString());receipt.setProperty("analysis",selfcheck?"material_check":"transient");receipt.setProperty("degree","4");receipt.setProperty("quadrature_order","20");receipt.setProperty("mesh_h_mm",f(h*1e3));receipt.setProperty("internal_step_us",f(step*1e6));receipt.setProperty("output_step_us",f(sample*1e6));receipt.setProperty("laws",laws.toString());receipt.setProperty("mph_solution_state","cleared");
  try(OutputStream out=Files.newOutputStream(Paths.get("completed.properties"))){receipt.store(out,"Actual COMSOL results; completion is not an accuracy certificate");}
 }
 private static Model materialCheck(Model m)throws Exception{
  double[][][] strains={{{.001,0,0},{0,0,0},{0,0,0}},{{0,.001,0},{.001,0,0},{0,0,0}},{{.001,.0004,-.0002},{.0004,-.0005,.0003},{-.0002,.0003,.0007}}};
  double[][][] expected={{{109.563,0,0},{0,57.957,0},{0,0,57.957}},{{-.594,52.2,0},{52.2,-.594,0},{0,0,-.249}},{{120.89172,20.71218,-10.30434},{20.71218,43.28142,15.57726},{-10.30434,15.57726,105.52146}}};
  m.study().create("stdStatic");m.study("stdStatic").create("stat","Stationary");m.study("stdStatic").createAutoSequences("all");String solver=m.sol().tags()[0];
  for(String tag:m.sol(solver).feature().tags())if(m.sol(solver).feature(tag).getType().equals("Stationary")){m.sol(solver).feature(tag).set("stol",1e-8);describe(m.sol(solver).feature(tag),"stationary");}
  try(PrintWriter out=new PrintWriter(Files.newBufferedWriter(Paths.get("coarse.csv")))){
   out.println("state,row,column,E_target,S_MPa,S_expected_MPa");
   for(int index=0;index<3;index++){
    double[][] chol=new double[3][3];for(int i=0;i<3;i++)for(int j=0;j<=i;j++){double x=(i==j?1:0)+2*strains[index][i][j];for(int k=0;k<j;k++)x-=chol[i][k]*chol[j][k];chol[i][j]=i==j?Math.sqrt(x):x/chol[j][j];}
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)m.param().set("d"+i+j,f(chol[j][i]-(i==j?1:0)));
    System.out.println("P1_MATERIAL_CHECK state="+index);long start=System.nanoTime();m.sol(solver).runAll();receipt.setProperty("state"+index+"_solve_seconds",f((System.nanoTime()-start)*1e-9));receipt.setProperty("dofs",Integer.toString(m.sol(solver).getU().length));
    String[] expressions=new String[18];for(int a=0;a<3;a++)for(int b=0;b<3;b++){expressions[3*a+b]="gs"+a+b+"/1e6";expressions[9+3*a+b]="abs(ge"+a+b+"-("+f(strains[index][a][b])+"))";}
    double[][] stress=evaluate(m,"stress","AvVolume",expressions,new int[]{1});double[][] error=evaluate(m,"strainError","MaxVolume",Arrays.copyOfRange(expressions,9,18),new int[]{1});
    for(int a=0;a<3;a++)for(int b=0;b<3;b++){double value=stress[3*a+b][0];if(!Double.isFinite(value)||Math.abs(value-expected[index][a][b])>1e-5||error[3*a+b][0]>1e-10)throw new IllegalStateException("Constitutive self-check failed");out.printf(Locale.ROOT,"%d,%d,%d,%.17g,%.17g,%.17g%n",index,a,b,strains[index][a][b],value,expected[index][a][b]);receipt.setProperty("state"+index+"_S"+a+b+"_MPa",f(value));}out.flush();m.sol(solver).clearSolutionData();
   }if(out.checkError())throw new IOException("Stress export");
  }
  timeSolver(m,false);finish(m,solver);return null;
 }
 private static double at(double[][][] data,int expr,int time,int point,int nt,int np){double value;if(data.length!=4)throw new IllegalStateException("Expressions");if(data[0].length==nt&&data[0][0].length==np)value=data[expr][time][point];else if(data[0].length==np&&data[0][0].length==nt)value=data[expr][point][time];else throw new IllegalStateException("Interpolation dimensions");if(!Double.isFinite(value))throw new IllegalStateException("Nonfinite receiver");return value;}
 private static Model transientRun(Model m)throws Exception{
  int nt=601,nr=13;int[] all=new int[nt];for(int i=0;i<nt;i++)all[i]=i+1;double[][] rx=new double[3][nr];for(int i=0;i<nr;i++){double[] point=vector(obj(receivers.get(i)).get("point_mm"));for(int a=0;a<3;a++)rx[a][i]=point[a]*1e-3;}
  String solver=timeSolver(m,true);
  try(PrintWriter out=new PrintWriter(Files.newBufferedWriter(Paths.get(level+".csv")));PrintWriter guards=new PrintWriter(Files.newBufferedWriter(Paths.get(level+".field.csv")))){
   out.println("law,rx,time_s,u_m,v_m,w_m");guards.println("law,time_s,max_green_strain,min_J");
   for(Object item:laws){int law=((Double)item).intValue();m.param().set("geom_on",law==0?"0":"1");m.param().set("mat_on",law==2?"1":"0");cacheConstantMatrices(m,solver,law);System.out.println("P1_SOLVE law="+law+" duration_us=30 h_mm="+f(h*1e3)+" step_us="+f(step*1e6));long start=System.nanoTime();m.sol(solver).runAll();receipt.setProperty("law"+law+"_solve_seconds",f((System.nanoTime()-start)*1e-9));receipt.setProperty("dofs",Integer.toString(m.sol(solver).getU().length));
    if(Arrays.asList(m.result().numerical().tags()).contains("receivers"))m.result().numerical().remove("receivers");m.result().numerical().create("receivers","Interp");m.result().numerical("receivers").set("expr",new String[]{"t","u","v","w"});m.result().numerical("receivers").set("data","dset1");m.result().numerical("receivers").set("edim",3);m.result().numerical("receivers").selection().all();m.result().numerical("receivers").set("coord",rx);m.result().numerical("receivers").set("solnum",all);double[][][] values=m.result().numerical("receivers").getData();
    for(int i=0;i<nt;i++)for(int r=0;r<nr;r++){double when=at(values,0,i,r,nt,nr);if(Math.abs(when-i*sample)>1e-15)throw new IllegalStateException("Output grid");out.printf(Locale.ROOT,"%d,%d,%.17g,%.17g,%.17g,%.17g%n",law,r,when,at(values,1,i,r,nt,nr),at(values,2,i,r,nt,nr),at(values,3,i,r,nt,nr));}out.flush();
    double[][] guard=evaluate(m,"guards","MaxVolume",new String[]{"sqrt(trE2)","-detF"},all);if(guard.length!=2||guard[0].length!=nt)throw new IllegalStateException("Guard times");double maxE=0,minJ=Double.POSITIVE_INFINITY;
    for(int i=0;i<nt;i++){double strain=guard[0][i],J=-guard[1][i];if(!Double.isFinite(strain)||!Double.isFinite(J)||strain>.001||J<=0)throw new IllegalStateException("Strain/J guard");maxE=Math.max(maxE,strain);minJ=Math.min(minJ,J);guards.printf(Locale.ROOT,"%d,%.17g,%.17g,%.17g%n",law,i*sample,strain,J);}guards.flush();receipt.setProperty("law"+law+"_max_green_strain",f(maxE));receipt.setProperty("law"+law+"_min_J",f(minJ));receipt.setProperty("guard_output_times","601");m.sol(solver).clearSolutionData();
   }if(out.checkError()||guards.checkError())throw new IOException("Export");
  }finish(m,solver);return null;
 }
 // Cache exact constant matrices only. Nonlinear stiffness and forcing are recomputed.
 private static void cacheConstantMatrices(Model m,String solver,int law){
  String time=null;for(String tag:m.sol(solver).feature().tags())if(m.sol(solver).feature(tag).getType().equals("Time"))time=tag;
  SolverFeature t=m.sol(solver).feature(time);String advanced=null;for(String tag:t.feature().tags())if(t.feature(tag).getType().equals("Advanced"))advanced=tag;
  if(advanced==null){advanced="constantMatrices";t.create(advanced,"Advanced");}SolverFeature a=t.feature(advanced);a.set("keep","on");for(String key:new String[]{"E","D","M","N"})a.set(key,"on");a.set("K",law==0?"on":"off");a.set("L","off");a.set("cachepattern",true);a.set("reusenullspacedata","on");
 }
 public static Model run()throws Exception{read();Model m=build();return selfcheck?materialCheck(m):transientRun(m);}
 public static void main(String[] args)throws Exception{run();}
}
