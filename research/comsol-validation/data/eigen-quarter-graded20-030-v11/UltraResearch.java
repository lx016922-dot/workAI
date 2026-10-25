// WorkAI independent COMSOL 3D reference. SI units; full nominal-traction weak form.
// This source is installed only through the existing signed model-revision API.
// V11 quarter-domain reference with explicit, audited one-sided cross-section grading.
// Material equations and existing time/export/element budgets are preserved.
import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
public class UltraResearch {
 private static Properties p,receipt;
 private static Map<String,Object> scene,cfg,geometry,material,tx,signal,acquisition;
 private static List<?> receivers,laws;
 private static int[] loadFaces,fixedFaces;
 private static String level,kind,solverMethod,meshType,analysis;
 private static double fullW,fullH;
 private static double L,W,H,h,step,duration,sample; private static int quadrature;
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
  scene=obj(root.get("scene"));cfg=obj(root.get("execution"));
  Set<String> required=new HashSet<>(Arrays.asList("level","coarse_h_mm","fine_h_mm","max_step_us","relative_tolerance","absolute_tolerance_m","quadrature_order","solver","laws","field_nx","field_ny","field_nz","field_frames"));if(!cfg.keySet().containsAll(required))throw new IllegalArgumentException("Missing execution fields");required.addAll(Arrays.asList("mesh_type","analysis","eigenmodes","sample_degree","sample_h_mm","artifact_encoding","grading_ratio","cross_section_grading_ratio","domain_symmetry"));if(!required.containsAll(cfg.keySet()))throw new IllegalArgumentException("Unknown execution fields");
  meshType=(String)cfg.getOrDefault("mesh_type","tet");analysis=(String)cfg.getOrDefault("analysis","transient");if(!Arrays.asList("tet","hex").contains(meshType)||!Arrays.asList("transient","eigen").contains(analysis))throw new IllegalArgumentException("Discretization/analysis");
  level=(String)cfg.get("level");if(!Arrays.asList("coarse","fine","timecheck").contains(level))throw new IllegalArgumentException("Output level");
  solverMethod=(String)cfg.get("solver");if(!Arrays.asList("genalpha","bdf","dopri5").contains(solverMethod))throw new IllegalArgumentException("Solver method");
  geometry=obj(scene.get("geometry"));material=obj(scene.get("material"));tx=obj(scene.get("source"));signal=obj(scene.get("signal"));acquisition=obj(scene.get("acquisition"));
  eq(scene.get("schema"),"workai.hyperelastic.solid.v1");eq(scene.get("constitutive"),"murnaghan");
  L=val(geometry,"length_mm",8,12)*1e-3;kind=(String)geometry.get("kind");if(!Arrays.asList("plate","pipe","hole","notch").contains(kind))throw new IllegalArgumentException("Geometry");
  W=val(geometry,"width_mm",3,4)*1e-3;H=val(geometry,"thickness_mm",1.5,2.1)*1e-3;
  String[] mk={"rho_kg_m3","lambda_pa","mu_pa","l_pa","m_pa","n_pa"};double[] mv={2700,58.1e9,26.1e9,-143e9,-297e9,-345e9};
  for(int j=0;j<mk.length;j++)close(num(material,mk[j]),mv[j],mk[j]);
  duration=val(acquisition,"duration_us",.1,5000)*1e-6;sample=val(acquisition,"sample_step_us",.025,.2)*1e-6;
  h=val(cfg,level.equals("coarse")?"coarse_h_mm":"fine_h_mm",.15,2.5)*1e-3;
  double maxstep=val(cfg,"max_step_us",.0001,.2)*1e-6;step=maxstep*(level.equals("timecheck")?.5:1);
  if(duration/(maxstep*.5)>100000.1||sample<maxstep||Math.abs(duration/sample-Math.rint(duration/sample))>1e-6)throw new IllegalArgumentException("Time/output budget");
  quadrature=(int)val(cfg,"quadrature_order",8,20);if(quadrature!=num(cfg,"quadrature_order"))throw new IllegalArgumentException("Quadrature integer");
  val(cfg,"relative_tolerance",1e-10,1e-6);val(cfg,"absolute_tolerance_m",1e-20,1e-14);
  laws=(List<?>)cfg.get("laws");Set<Integer> used=new HashSet<>();for(Object law:laws){double d=(Double)law;int k=(int)d;if(k!=d||k<0||k>2||!used.add(k))throw new IllegalArgumentException("Controls");}if(laws.size()<1)throw new IllegalArgumentException("No controls");
  receivers=(List<?>)scene.get("receivers");if(receivers.size()<1||receivers.size()>8)throw new IllegalArgumentException("Receivers");
  for(Object r:receivers){double[] v=vector(obj(r).get("point_mm"));for(double x:v)if(!Double.isFinite(x))throw new IllegalArgumentException("Receiver");}
  val(tx,"amplitude_mpa",.6,1.5);val(tx,"frequency_mhz",.18,.28);val(tx,"cycles",1,20);val(tx,"delay_us",.05,.5);val(tx,"phase_deg",-75,75);
  String excitation=(String)signal.get("kind");if(!Arrays.asList("hann","gaussian","rectangular","chirp").contains(excitation))throw new IllegalArgumentException("Excitation");val(signal,"end_frequency_mhz",.18,.28);
  if(!Arrays.asList("xmin","xmax").contains(tx.get("face")))throw new IllegalArgumentException("Traction face");
  if(!Arrays.asList("none","xmin","xmax").contains(obj(scene.get("boundary")).get("fixed_face")))throw new IllegalArgumentException("Fixed face");
  // The legacy boundary token describes natural (non-absorbing) boundaries;
  // the explicit signed 3D fixed_face adds an exact essential constraint.
  eq(p.getProperty("schema"),"workai.comsol.research.case.v1");eq(p.getProperty("boundary"),"free");close(n("axisymmetric"),0,"axisymmetric");close(n("thermal"),0,"thermal");
  close(n("length_mm"),L*1e3,"length");close(n("height_mm"),H*1e3,"thickness");close(n("rho"),num(material,"rho_kg_m3"),"rho");
  close(n("cl"),Math.sqrt((num(material,"lambda_pa")+2*num(material,"mu_pa"))/num(material,"rho_kg_m3")),"cl");close(n("ct"),Math.sqrt(num(material,"mu_pa")/num(material,"rho_kg_m3")),"ct");
  for(String k:new String[]{"duration_us","sample_step_us"})close(n(k),num(acquisition,k),k);
  for(String k:new String[]{"coarse_h_mm","fine_h_mm","max_step_us","relative_tolerance","field_nx","field_ny","field_frames"})close(n(k),num(cfg,k),k);
  close(n("shot_count"),laws.size(),"shots");close(n("receiver_count"),receivers.size(),"receivers");close(n("element_order"),4,"degree");
  for(int j=0;j<receivers.size();j++)close(n("rx"+j+"_x_mm"),vector(obj(receivers.get(j)).get("point_mm"))[0],"receiver x");
  if(laws.size()*receivers.size()*(Math.round(duration/sample)+1)>2000000)throw new IllegalArgumentException("Probe export budget");
  if(laws.size()*num(cfg,"field_nx")*num(cfg,"field_ny")*num(cfg,"field_nz")*num(cfg,"field_frames")>2000000)throw new IllegalArgumentException("3D field export budget");
  if(Files.exists(Paths.get("completed.properties")))throw new IllegalArgumentException("Completed job is immutable");
  receipt=new Properties();for(String k:new String[]{"job_id","scene_sha256","source_sha256","parameters_sha256"})receipt.setProperty(k,p.getProperty(k));receipt.setProperty("case_id",(String)root.get("case_id"));
 }
 private static int[] box(Model m,String tag,double x){m.component("comp1").selection().create(tag,"Box");m.component("comp1").selection(tag).set("entitydim",2);m.component("comp1").selection(tag).set("xmin",x-1e-8);m.component("comp1").selection(tag).set("xmax",x+1e-8);m.component("comp1").selection(tag).set("ymin",-.1);m.component("comp1").selection(tag).set("ymax",.1);m.component("comp1").selection(tag).set("zmin",-.1);m.component("comp1").selection(tag).set("zmax",.1);m.component("comp1").selection(tag).set("condition","allvertices");int[] ids=m.component("comp1").selection(tag).entities(2);if(ids.length==0)throw new IllegalStateException("Empty boundary selection: "+tag);System.out.println("WORKAI_BOUNDARY "+tag+"="+Arrays.toString(ids));receipt.setProperty(tag+"_boundaries",Arrays.toString(ids));return ids;}
 private static String pulse(){
  double f0=num(tx,"frequency_mhz")*1e6,f1="chirp".equals(signal.get("kind"))?num(signal,"end_frequency_mhz")*1e6:f0,T=num(tx,"cycles")/((f0+f1)/2),delay=num(tx,"delay_us")*1e-6;
  String time="(t-"+f(delay)+")",envelope="1";
  if(Arrays.asList("hann","chirp").contains(signal.get("kind")))envelope="0.5*(1-cos(2*pi*"+time+"/"+f(T)+"))";
  if("gaussian".equals(signal.get("kind")))envelope="exp(-0.5*(("+time+"-"+f(T/2)+")/"+f(T/6)+")^2)";
  return "if((t>="+f(delay)+")&&(t<="+f(delay+T)+"),"+f(num(tx,"amplitude_mpa")*1e6)+"*"+envelope+"*sin(2*pi*("+f(f0)+"*"+time+"+"+f((f1-f0)/(2*T))+"*"+time+"^2)+"+f(num(tx,"phase_deg")*Math.PI/180)+"),0)";
 }
 private static void variable(Model m,String key,String expression){m.variable("vars").set(key,expression);}
 private static Model build(){
  if(!kind.equals("plate")||!meshType.equals("hex")||!"quarter_yz".equals(cfg.get("domain_symmetry")))throw new IllegalArgumentException("Quarter reference scope");
  double[] axial=vector(tx.get("direction"));if(axial[0]==0||axial[1]!=0||axial[2]!=0||num(tx,"radius_mm")!=0)throw new IllegalArgumentException("Quarter symmetry requires uniform axial traction");
  fullW=W;fullH=H;W*=.5;H*=.5;
  receipt.setProperty("domain_symmetry","quarter_yz");receipt.setProperty("physical_width_m",f(fullW));receipt.setProperty("physical_thickness_m",f(fullH));receipt.setProperty("physical_domain_multiplicity","4");receipt.setProperty("cross_section_grading",cfg.containsKey("cross_section_grading_ratio")?"one_sided_outer_surface":"uniform");receipt.setProperty("cross_section_grading_ratio",String.valueOf(cfg.getOrDefault("cross_section_grading_ratio",1.0)));
  Model m=ModelUtil.create("GraphWaveReference");m.label("Independent three-dimensional Murnaghan reference");
  m.component().create("comp1");m.component("comp1").geom().create("geom1",3);
  if(kind.equals("pipe")){
   for(String tag:new String[]{"body","void"}){m.component("comp1").geom("geom1").create(tag,"Cylinder");m.component("comp1").geom("geom1").feature(tag).set("axis",new double[]{1,0,0});m.component("comp1").geom("geom1").feature(tag).set("h",L);m.component("comp1").geom("geom1").feature(tag).set("r",num(geometry,tag.equals("body")?"outer_radius_mm":"inner_radius_mm")*1e-3);}
  }else{
   m.component("comp1").geom("geom1").create("body","Block");m.component("comp1").geom("geom1").feature("body").set("size",new double[]{L,W,H});
   if(kind.equals("hole")){m.component("comp1").geom("geom1").create("void","Cylinder");m.component("comp1").geom("geom1").feature("void").set("r",num(geometry,"hole_radius_mm")*1e-3);m.component("comp1").geom("geom1").feature("void").set("h",H);m.component("comp1").geom("geom1").feature("void").set("pos",new double[]{num(geometry,"hole_x_mm")*1e-3,num(geometry,"hole_y_mm")*1e-3,0});}
   if(kind.equals("notch")){double width=num(geometry,"notch_width_mm")*1e-3,depth=num(geometry,"notch_depth_mm")*1e-3;m.component("comp1").geom("geom1").create("void","Block");m.component("comp1").geom("geom1").feature("void").set("size",new double[]{width,W,depth});m.component("comp1").geom("geom1").feature("void").set("pos",new double[]{num(geometry,"notch_x_mm")*1e-3-width/2,0,H-depth});}
  }
  if(!kind.equals("plate")){m.component("comp1").geom("geom1").create("cut","Difference");m.component("comp1").geom("geom1").feature("cut").selection("input").set(new String[]{"body"});m.component("comp1").geom("geom1").feature("cut").selection("input2").set(new String[]{"void"});}
  m.component("comp1").geom("geom1").run();m.component("comp1").mesh().create("mesh1","geom1");m.component("comp1").mesh("mesh1").feature("size").set("custom","on");m.component("comp1").mesh("mesh1").feature("size").set("hmax",h);m.component("comp1").mesh("mesh1").feature("size").set("hmin",h/3);m.component("comp1").mesh("mesh1").feature("size").set("hgrad",1.3);if(meshType.equals("hex")){
   if(!kind.equals("plate"))throw new IllegalArgumentException("Mapped hex reference currently requires an exact rectangular plate");
   int[] start=box(m,"meshStart",0),end=box(m,"meshEnd",L);MeshSequence mesh=m.component("comp1").mesh("mesh1");mesh.create("map","Map");mesh.feature("map").selection().set(start);
   for(int axis=1;axis<3;axis++){String tag="edge"+axis;m.component("comp1").selection().create(tag,"Box");m.component("comp1").selection(tag).set("entitydim",1);String[] dim={"x","y","z"};double[] size={L,W,H};for(int k=0;k<3;k++){m.component("comp1").selection(tag).set(dim[k]+"min",-1e-8);m.component("comp1").selection(tag).set(dim[k]+"max",k==axis?size[k]+1e-8:1e-8);}m.component("comp1").selection(tag).set("condition","allvertices");int[] ids=m.component("comp1").selection(tag).entities(1);if(ids.length!=1)throw new IllegalStateException("Mapped edge selection");mesh.feature("map").create(tag,"Distribution");ids=gradedEdges(m,axis,ids,size);mesh.feature("map").feature(tag).selection().set(ids);mesh.feature("map").feature(tag).set("numelem",(int)Math.ceil(size[axis]/h));mesh.feature("map").feature(tag).set("equidistant","on");grade(mesh.feature("map").feature(tag),true);}
   mesh.create("sweep","Sweep");mesh.feature("sweep").selection("sourceface").set(start);mesh.feature("sweep").selection("targetface").set(end);mesh.feature("sweep").create("layers","Distribution");mesh.feature("sweep").feature("layers").selection().all();mesh.feature("sweep").feature("layers").set("numelem",(int)Math.ceil(L/h));mesh.feature("sweep").feature("layers").set("equidistant","on");grade(mesh.feature("sweep").feature("layers"),false);
  }else m.component("comp1").mesh("mesh1").create("tet","FreeTet");m.component("comp1").mesh("mesh1").run();receipt.setProperty("mesh_type",meshType);
  double[][] vertices=m.component("comp1").mesh("mesh1").getVertex();for(int axis=0;axis<3;axis++){TreeSet<Double> positions=new TreeSet<>();for(double coordinate:vertices[axis])positions.add(coordinate);receipt.setProperty("mesh_axis_"+axis+"_m",positions.toString());}receipt.setProperty("grading_ratio",String.valueOf(cfg.getOrDefault("grading_ratio",1.0)));
  int ne=m.component("comp1").mesh("mesh1").getNumElem();if(ne>n("element_budget"))throw new IllegalArgumentException("Actual element budget exceeded before solving");receipt.setProperty("elements",Integer.toString(ne));
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
  StringJoiner bulk=new StringJoiner("+","rho*(utt*test(u)+vtt*test(v)+wtt*test(w))+","");
  for(int a=0;a<3;a++)for(int b=0;b<3;b++){StringJoiner sum=new StringJoiner("+");for(int k=0;k<3;k++)sum.add("gf"+a+k+"*gs"+k+b);variable(m,"gp"+a+b,"if(geom_on>0.5,"+sum+","+(a==b?"lam*trL+":"")+"2*mu*gl"+a+b+")");bulk.add("gp"+a+b+"*test("+vars[a]+dirs[b]+")");}
  variable(m,"energyW","if(geom_on>0.5,0.5*lam*trE^2+mu*trE2+mat_on*(aa/3*trE3+bb*trE*trE2+cc/3*trE^3),0.5*lam*trL^2+mu*trL2)");variable(m,"energyK","0.5*rho*(ut^2+vt^2+wt^2)");variable(m,"drive",pulse());
  m.weak().create("bulk");m.weak("bulk").selection().geom("geom1",3).all();m.weak("bulk").intRule("quad");m.weak("bulk").weak(bulk.toString());
  loadFaces=box(m,"load","xmin".equals(tx.get("face"))?0:L);double[] direction=vector(tx.get("direction"));double norm=Math.sqrt(direction[0]*direction[0]+direction[1]*direction[1]+direction[2]*direction[2]);if(norm<1e-14)throw new IllegalArgumentException("Zero traction direction");
  StringJoiner test=new StringJoiner("+"),power=new StringJoiner("+");for(int j=0;j<3;j++){test.add(f(direction[j]/norm)+"*test("+vars[j]+")");power.add(f(direction[j]/norm)+"*"+vars[j]+"t");}
  variable(m,"drivePower","drive*("+power+")");m.weak().create("load");m.weak("load").selection().geom("geom1",2).set(loadFaces);m.weak("load").intRule("quad");m.weak("load").weak("-drive*("+test+")");
  String fixed=(String)obj(scene.get("boundary")).get("fixed_face");if(!fixed.equals("none")){fixedFaces=box(m,"fixed",fixed.equals("xmin")?0:L);m.constr().create("fixed",new String[]{"shape_u","shape_v","shape_w"});ConstrFeature c=m.constr("fixed").create("zero");c.set("constr",new String[]{"u","v","w"});c.selection().geom("geom1",2).set(fixedFaces);}
  for(int axis=1;axis<3;axis++){
   String tag=axis==1?"symmetryY":"symmetryZ",component=axis==1?"v":"w",coordinate=axis==1?"y":"z";double plane=axis==1?W:H;
   m.component("comp1").selection().create(tag,"Box");m.component("comp1").selection(tag).set("entitydim",2);
   for(String dimension:new String[]{"x","y","z"}){m.component("comp1").selection(tag).set(dimension+"min",dimension.equals(coordinate)?plane-1e-8:-.1);m.component("comp1").selection(tag).set(dimension+"max",dimension.equals(coordinate)?plane+1e-8:.1);}
   m.component("comp1").selection(tag).set("condition","allvertices");int[] faces=m.component("comp1").selection(tag).entities(2);if(faces.length!=1)throw new IllegalStateException("Unique reflection plane required");
   m.constr().create(tag,new String[]{"shape_"+component});ConstrFeature constraint=m.constr(tag).create("zero");constraint.set("constr",new String[]{component});constraint.selection().geom("geom1",2).set(faces);receipt.setProperty(tag+"_boundaries",Arrays.toString(faces));
  }
  return m;
 }
 private static double[][][] interpolate(Model m,String tag,double[][] coordinates,int[] solutions){
  double[][] folded=new double[3][coordinates[0].length];for(int point=0;point<coordinates[0].length;point++){folded[0][point]=coordinates[0][point];folded[1][point]=coordinates[1][point]>W?fullW-coordinates[1][point]:coordinates[1][point];folded[2][point]=coordinates[2][point]>H?fullH-coordinates[2][point]:coordinates[2][point];}
  if(Arrays.asList(m.result().numerical().tags()).contains(tag))m.result().numerical().remove(tag);
  m.result().numerical().create(tag,"Interp");m.result().numerical(tag).set("expr",new String[]{analysis.equals("eigen")?"freq":"t","u","v","w"});m.result().numerical(tag).set("data","dset1");m.result().numerical(tag).set("edim",3);m.result().numerical(tag).selection().all();m.result().numerical(tag).set("coord",folded);m.result().numerical(tag).set("solnum",solutions);double[][][] values=m.result().numerical(tag).getData();
  int nt=solutions.length,np=coordinates[0].length;for(int expr=2;expr<4;expr++)for(int point=0;point<np;point++){
   if(coordinates[expr-1][point]>(expr==2?W:H))for(int time=0;time<nt;time++){
    if(values[0].length==nt&&values[0][0].length==np)values[expr][time][point]*=-1;
    else if(values[0].length==np&&values[0][0].length==nt)values[expr][point][time]*=-1;
    else throw new IllegalStateException("Reflected interpolation dimensions");
   }
  }return values;
 }
 private static double at(double[][][] a,int expr,int t,int point,int nt,int np){double v;if(a.length!=4)throw new IllegalStateException("Expressions");if(a[0].length==nt&&a[0][0].length==np)v=a[expr][t][point];else if(a[0].length==np&&a[0][0].length==nt)v=a[expr][point][t];else throw new IllegalStateException("Interpolation dimensions");if(!Double.isFinite(v))throw new IllegalStateException("Nonfinite output/outside receiver");return v;}
 private static double[][] integral(Model m,String tag,String type,String[] expressions,int[] solutions,String selection){
  if(Arrays.asList(m.result().numerical().tags()).contains(tag))m.result().numerical().remove(tag);m.result().numerical().create(tag,type);if(type.startsWith("Int")){m.result().numerical(tag).set("intorderactive","on");m.result().numerical(tag).set("intorder",quadrature);}m.result().numerical(tag).set("expr",expressions);m.result().numerical(tag).set("data","dset1");if(selection==null)m.result().numerical(tag).selection().all();else m.result().numerical(tag).selection().set(selection.equals("load")?loadFaces:fixedFaces);m.result().numerical(tag).set("solnum",solutions);return m.result().numerical(tag).getReal();
 }
 public static Model run()throws Exception{
  read();Model m=build();if(analysis.equals("eigen"))return eigen(m);int nt=(int)Math.round(duration/sample)+1,nr=receivers.size(),nf=(int)num(cfg,"field_frames");int[] all=new int[nt],frames=new int[nf];for(int j=0;j<nt;j++)all[j]=j+1;for(int j=0;j<nf;j++)frames[j]=(int)Math.round((nt-1.)*j/(nf-1))+1;
  double[][] rx=new double[3][nr];for(int j=0;j<nr;j++){double[] r=vector(obj(receivers.get(j)).get("point_mm"));for(int k=0;k<3;k++)rx[k][j]=r[k]*1e-3;}
  String times="range(0,"+f(sample)+","+f(duration)+")";m.study().create("std1");m.study("std1").create("time","Transient");m.study("std1").feature("time").set("tlist",times);m.study("std1").createAutoSequences("all");String solver=m.sol().tags()[0],time=null;
  for(String tag:m.sol(solver).feature().tags())if(m.sol(solver).feature(tag).getType().equals("Time"))time=tag;if(time==null)throw new IllegalStateException("Time solver missing");
  m.sol(solver).feature(time).set("tlist",times);m.sol(solver).feature(time).set("rtol",num(cfg,"relative_tolerance"));m.sol(solver).feature(time).set("atolglobalvaluemethod","manual");m.sol(solver).feature(time).set("atolglobalmethod","unscaled");m.sol(solver).feature(time).set("atolglobal",num(cfg,"absolute_tolerance_m"));m.sol(solver).feature(time).set("reacf","off");
  if(solverMethod.equals("dopri5")){m.sol(solver).feature(time).set("odesolvertype","explicit");m.sol(solver).feature(time).set("timemethodexp","rk");m.sol(solver).feature(time).set("rkmethod","dopri5");m.sol(solver).feature(time).set("tstepsdopri5","manual");m.sol(solver).feature(time).set("timestepdopri5",f(step));}else{m.sol(solver).feature(time).set("timemethod",solverMethod);if(solverMethod.equals("genalpha")){m.sol(solver).feature(time).set("tstepsgenalpha","manual");m.sol(solver).feature(time).set("timestepgenalpha",f(step));m.sol(solver).feature(time).set("rhoinf",1.0);}else{m.sol(solver).feature(time).set("tstepsbdf","manual");m.sol(solver).feature(time).set("timestepbdf",f(step));m.sol(solver).feature(time).set("maxorder",5);m.sol(solver).feature(time).set("bdforder","5");}}
  try(PrintWriter out=new PrintWriter(Files.newBufferedWriter(Paths.get(level+".csv")));PrintWriter energy=new PrintWriter(Files.newBufferedWriter(Paths.get(level+".field.csv")))){
   out.println("law,rx,time_s,u_m,v_m,w_m");energy.println("law,time_s,kinetic_j,potential_j,traction_power_w,integrated_work_j");
   for(Object value:laws){int law=((Double)value).intValue();m.param().set("geom_on",law==0?"0":"1");m.param().set("mat_on",law==2?"1":"0");cacheConstantMatrices(m,solver,time,law);System.out.println("WORKAI_COMSOL_3D_SOLVE level="+level+" law="+law+" duration_us="+f(duration*1e6));long start=System.nanoTime();m.sol(solver).runAll();receipt.setProperty("law"+law+"_solve_seconds",f((System.nanoTime()-start)*1e-9));
    double[][][] values=interpolate(m,"receivers",rx,all);for(int t=0;t<nt;t++)for(int r=0;r<nr;r++){double when=at(values,0,t,r,nt,nr);if(Math.abs(when-t*sample)>1e-11)throw new IllegalStateException("Output time grid");out.printf(Locale.ROOT,"%d,%d,%.17g,%.17g,%.17g,%.17g%n",law,r,when,at(values,1,t,r,nt,nr),at(values,2,t,r,nt,nr),at(values,3,t,r,nt,nr));}out.flush();
    double[][] en=integral(m,"energy","IntVolume",new String[]{"energyK","energyW"},all,null),power=integral(m,"power","IntSurface",new String[]{"drivePower"},all,"load");if(en.length!=2||power.length!=1||en[0].length!=nt||power[0].length!=nt)throw new IllegalStateException("Integral dimensions");double work=0,scale=0,error=0;
    for(int t=0;t<nt;t++){if(t>0)work+=sample*(power[0][t]+power[0][t-1])/2;double total=en[0][t]+en[1][t];if(!Double.isFinite(total)||!Double.isFinite(work))throw new IllegalStateException("Energy finite");scale=Math.max(scale,Math.max(Math.abs(total),Math.abs(work)));error=Math.max(error,Math.abs(total-work));energy.printf(Locale.ROOT,"%d,%.17g,%.17g,%.17g,%.17g,%.17g%n",law,t*sample,en[0][t],en[1][t],power[0][t],work);}energy.flush();receipt.setProperty("law"+law+"_sampled_work_balance_relative",f(error/Math.max(scale,1e-30)));
    double[][] guards=integral(m,"guard","MaxVolume",new String[]{"sqrt(trE2)","-detF"},frames,null);double strain=0,minJ=Double.POSITIVE_INFINITY;for(double v:guards[0])strain=Math.max(strain,v);for(double v:guards[1])minJ=Math.min(minJ,-v);if(!Double.isFinite(strain)||!Double.isFinite(minJ)||strain>.001||minJ<=0)throw new IllegalStateException("Strain/J guard failed");receipt.setProperty("law"+law+"_sampled_max_strain",f(strain));receipt.setProperty("law"+law+"_sampled_min_J",f(minJ));
    if(!obj(scene.get("boundary")).get("fixed_face").equals("none")){double[][] fixed=integral(m,"fixedCheck","MaxSurface",new String[]{"sqrt(u^2+v^2+w^2)"},frames,"fixed");double worst=0;for(double v:fixed[0])worst=Math.max(worst,Math.abs(v));if(!Double.isFinite(worst)||worst>1e-18)throw new IllegalStateException("Fixed face moved");receipt.setProperty("law"+law+"_sampled_fixed_displacement_m",f(worst));}
    m.sol(solver).clearSolutionData();
   }if(out.checkError()||energy.checkError())throw new IOException("Export write failure");
  }
  encodeCsv(level+".csv");encodeCsv(level+".field.csv");m.save(level+".mph");Model reopened=ModelUtil.loadCopy("Reopened",level+".mph");if(reopened.sol().tags().length==0||!reopened.sol(reopened.sol().tags()[0]).isEmpty())throw new IllegalStateException("Cleared model reopen failed");ModelUtil.remove(reopened.tag());ModelUtil.remove(m.tag());
  receipt.setProperty("schema","workai.comsol.graphwave3d.receipt.v1");receipt.setProperty("method","comsol_3d_reference_weakform_full_murnaghan");receipt.setProperty("status","completed");receipt.setProperty("comsol_version",ModelUtil.getComsolVersion());receipt.setProperty("completed_at_utc",java.time.Instant.now().toString());receipt.setProperty("level",level);receipt.setProperty("duration_us",f(duration*1e6));receipt.setProperty("sample_step_us",f(sample*1e6));receipt.setProperty("samples",Integer.toString(nt));receipt.setProperty("mesh_h_mm",f(h*1e3));receipt.setProperty("internal_step_us",f(step*1e6));receipt.setProperty("element_order","4");receipt.setProperty("quadrature_order",Integer.toString(quadrature));receipt.setProperty("solver",solverMethod);receipt.setProperty("guard_frames",Integer.toString(nf));receipt.setProperty("laws",laws.toString());receipt.setProperty("natural_boundaries","free_except_explicit_fixed_face");receipt.setProperty("fixed_face",(String)obj(scene.get("boundary")).get("fixed_face"));receipt.setProperty("power_integration","trapezoidal_on_complete_output_grid");receipt.setProperty("field_csv_role","complete_integrated_energy_and_external_power_histories");receipt.setProperty("mph_solution_state","cleared");receipt.setProperty("mph_reopen_verified","true");
  try(OutputStream out=Files.newOutputStream(Paths.get("completed.properties"))){receipt.store(out,"Actual COMSOL 3D results; no accuracy certificate from completion alone");}return null;
 }


 private static void encodeCsv(String name)throws Exception{
  String encoding=(String)cfg.getOrDefault("artifact_encoding","csv");if(encoding.equals("csv"))return;if(!encoding.equals("gzip_csv"))throw new IllegalArgumentException("Artifact encoding");
  byte[] raw=Files.readAllBytes(Paths.get(name));if(raw.length>268435456)throw new IllegalArgumentException("Decoded export exceeds 256 MiB");java.io.ByteArrayOutputStream compressed=new java.io.ByteArrayOutputStream();try(java.util.zip.GZIPOutputStream gzip=new java.util.zip.GZIPOutputStream(compressed)){gzip.write(raw);}
  String hash=sha(new String(raw,java.nio.charset.StandardCharsets.UTF_8));String header="encoding,decoded_bytes,decoded_sha256,payload_base64\n";String row="workai_gzip_csv_v1,"+raw.length+","+hash+","+Base64.getEncoder().encodeToString(compressed.toByteArray())+"\n";
  Files.write(Paths.get(name),(header+row).getBytes(java.nio.charset.StandardCharsets.US_ASCII));receipt.setProperty(name+"_encoding","workai_gzip_csv_v1");receipt.setProperty(name+"_decoded_bytes",Integer.toString(raw.length));receipt.setProperty(name+"_decoded_sha256",hash);
 }
 private static int[] gradedEdges(Model m,int axis,int[] first,double[] size){
  if(!cfg.containsKey("cross_section_grading_ratio"))return first;
  int other=3-axis;String tag="edge"+axis+"Opposite";String[] dim={"x","y","z"};
  m.component("comp1").selection().create(tag,"Box");m.component("comp1").selection(tag).set("entitydim",1);
  for(int k=0;k<3;k++){
   double lower=k==other?size[k]:0,upper=k==axis||k==other?size[k]:0;
   m.component("comp1").selection(tag).set(dim[k]+"min",lower-1e-8);
   m.component("comp1").selection(tag).set(dim[k]+"max",upper+1e-8);
  }
  m.component("comp1").selection(tag).set("condition","allvertices");
  int[] opposite=m.component("comp1").selection(tag).entities(1);
  if(first.length!=1||opposite.length!=1||first[0]==opposite[0])throw new IllegalStateException("Opposite mapped edges");
  return new int[]{first[0],opposite[0]};
 }
 private static void grade(MeshFeature feature,boolean symmetric){
  String key=symmetric?"cross_section_grading_ratio":"grading_ratio";
  if(!cfg.containsKey(key))return;double ratio=val(cfg,key,1,100);
  if(!meshType.equals("hex"))throw new IllegalArgumentException("Grading only for quarter hex references");
  int count=feature.getInt("numelem");
  // COMSOL Distribution API: predefined distributions use elemcount and
  // method=geometric (not numelem or a distrib property).
  feature.set("equidistant","off");feature.set("type","predefined");feature.set("elemcount",count);feature.set("method","geometric");feature.set("elemratio",ratio);feature.set("symmetric","off");feature.set("reverse",symmetric?"off":"on");
 }
 private static double[] gllAxis(double length,double size){
  int count=(int)Math.ceil(length/size);double[] local={-1,-Math.sqrt(3./7),0,Math.sqrt(3./7),1},axis=new double[4*count+1];for(int e=0;e<count;e++)for(int j=0;j<4;j++)axis[4*e+j]=length/count*(e+(local[j]+1)/2);axis[4*count]=length;return axis;
 }
 private static Model eigen(Model m)throws Exception{
  if(!kind.equals("plate")||laws.size()!=1||((Double)laws.get(0))!=0||num(cfg,"sample_degree")!=4)throw new IllegalArgumentException("Eigen reference scope");
  int modes=(int)val(cfg,"eigenmodes",8,200);if(modes!=num(cfg,"eigenmodes"))throw new IllegalArgumentException("Integer mode count");double size=val(cfg,"sample_h_mm",.8,2.5)*1e-3;
  double[][] axes={gllAxis(L,size),gllAxis(fullW,size),gllAxis(fullH,size)};int gridCount=axes[0].length*axes[1].length*axes[2].length,count=gridCount+receivers.size();if((long)count*modes>2000000)throw new IllegalArgumentException("Eigen export budget");
  double[][] points=new double[3][count];int index=0;for(double x:axes[0])for(double y:axes[1])for(double z:axes[2]){points[0][index]=x;points[1][index]=y;points[2][index++]=z;}
  for(Object receiver:receivers){double[] point=vector(obj(receiver).get("point_mm"));for(int j=0;j<3;j++)points[j][index]=point[j]*1e-3;index++;}receipt.setProperty("grid_sample_points",Integer.toString(gridCount));receipt.setProperty("receiver_sample_points",Integer.toString(receivers.size()));
  m.param().set("geom_on","0");m.param().set("mat_on","0");variable(m,"drive","0");
  m.study().create("std1");m.study("std1").create("eigen","Eigenfrequency");m.study("std1").feature("eigen").set("neigsactive","on");m.study("std1").feature("eigen").set("neigs",modes);m.study("std1").feature("eigen").set("shiftactive","on");m.study("std1").feature("eigen").set("shift","0");m.study("std1").createAutoSequences("all");String solver=m.sol().tags()[0];
  for(String tag:m.sol(solver).feature().tags())if(m.sol(solver).feature(tag).getType().equals("Eigenvalue")){m.sol(solver).feature(tag).set("rtol",1e-11);m.sol(solver).feature(tag).set("eigreschk",true);}
  System.out.println("WORKAI_COMSOL_EIGEN modes="+modes+" mesh="+meshType+" h_mm="+f(h*1e3));long start=System.nanoTime();m.sol(solver).runAll();receipt.setProperty("solve_seconds",f((System.nanoTime()-start)*1e-9));int[] solutions=new int[modes];for(int j=0;j<modes;j++)solutions[j]=j+1;
  double[][][] values=interpolate(m,"modes",points,solutions);double maxfreq=0;
  try(PrintWriter out=new PrintWriter(Files.newBufferedWriter(Paths.get(level+".csv")))){out.println("mode,point,x_m,y_m,z_m,frequency_hz,u_m,v_m,w_m");for(int mode=0;mode<modes;mode++)for(int point=0;point<count;point++){double frequency=at(values,0,mode,point,modes,count);if(frequency<=0)throw new IllegalStateException("Nonpositive eigenfrequency");maxfreq=Math.max(maxfreq,frequency);out.printf(Locale.ROOT,"%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g%n",mode,point,points[0][point],points[1][point],points[2][point],frequency,at(values,1,mode,point,modes,count),at(values,2,mode,point,modes,count),at(values,3,mode,point,modes,count));}if(out.checkError())throw new IOException("Modal export");}
  double[][] mass=integral(m,"modalMass","IntVolume",new String[]{"rho*(u^2+v^2+w^2)","2*energyW"},solutions,null),coupling=integral(m,"modalForce","IntSurface",new String[]{"u","v","w"},solutions,"load");
  try(PrintWriter out=new PrintWriter(Files.newBufferedWriter(Paths.get(level+".field.csv")))){out.println("mode,mass_integral,stiffness_integral,force_x_integral,force_y_integral,force_z_integral");for(int j=0;j<modes;j++){if(!(mass[0][j]>0)||!(mass[1][j]>0))throw new IllegalStateException("Nonpositive modal energy");out.printf(Locale.ROOT,"%d,%.17g,%.17g,%.17g,%.17g,%.17g%n",j,mass[0][j],mass[1][j],coupling[0][j],coupling[1][j],coupling[2][j]);}if(out.checkError())throw new IOException("Modal integrals export");}
  if(fixedFaces!=null){double[][] fixed=integral(m,"fixedCheck","MaxSurface",new String[]{"sqrt(u^2+v^2+w^2)"},solutions,"fixed");double worst=0;for(double x:fixed[0])worst=Math.max(worst,Math.abs(x));if(!Double.isFinite(worst)||worst>1e-12)throw new IllegalStateException("Fixed eigenmode boundary");receipt.setProperty("fixed_displacement_m",f(worst));}
  encodeCsv(level+".csv");encodeCsv(level+".field.csv");m.sol(solver).clearSolutionData();m.save(level+".mph");Model reopened=ModelUtil.loadCopy("Reopened",level+".mph");if(!reopened.sol(reopened.sol().tags()[0]).isEmpty())throw new IllegalStateException("Modal model reopen");ModelUtil.remove(reopened.tag());ModelUtil.remove(m.tag());
  receipt.setProperty("schema","workai.comsol.graphwave3d.eigen_receipt.v1");receipt.setProperty("method","comsol_3d_linear_eigenfrequency_weakform");receipt.setProperty("status","completed");receipt.setProperty("comsol_version",ModelUtil.getComsolVersion());receipt.setProperty("completed_at_utc",java.time.Instant.now().toString());receipt.setProperty("level",level);receipt.setProperty("modes",Integer.toString(modes));receipt.setProperty("sample_points",Integer.toString(count));receipt.setProperty("max_frequency_hz",f(maxfreq));receipt.setProperty("mesh_h_mm",f(h*1e3));receipt.setProperty("element_order","4");receipt.setProperty("quadrature_order",Integer.toString(quadrature));receipt.setProperty("mph_solution_state","cleared");receipt.setProperty("mph_reopen_verified","true");receipt.setProperty("scope","Quarter-domain axial-symmetry COMSOL eigenpairs reflected onto the full physical sampling grid; not a nonlinear 5 ms transient certificate");try(OutputStream out=Files.newOutputStream(Paths.get("completed.properties"))){receipt.store(out,"Actual COMSOL linear eigenfrequency reference");}return null;
 }

 // Reuse only matrices that are mathematically constant for this fixed mesh.
 // The load is time dependent, and nonlinear stiffness is never frozen.
 private static void cacheConstantMatrices(Model m,String solver,String time,int law){
  String advanced=null;for(String tag:m.sol(solver).feature(time).feature().tags())if(m.sol(solver).feature(time).feature(tag).getType().equals("Advanced"))advanced=tag;
  if(advanced==null){advanced="constantMatrices";m.sol(solver).feature(time).create(advanced,"Advanced");}
  m.sol(solver).feature(time).feature(advanced).set("keep","on");
  for(String property:new String[]{"E","D","M","N"})m.sol(solver).feature(time).feature(advanced).set(property,"on");
  m.sol(solver).feature(time).feature(advanced).set("K",law==0?"on":"off");m.sol(solver).feature(time).feature(advanced).set("L","off");
  m.sol(solver).feature(time).feature(advanced).set("cachepattern",true);m.sol(solver).feature(time).feature(advanced).set("reusenullspacedata","on");
 }
 public static void main(String[] args)throws Exception{run();}
}
