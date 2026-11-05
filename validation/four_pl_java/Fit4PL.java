// One bounded 4PL ML fit with meyerjp3/psychometrics (Apache-2.0) on a user-supplied GH grid.
// java Fit4PL data.csv grid.csv out.csv
import com.itemanalysis.psychometrics.irt.estimation.*;
import com.itemanalysis.psychometrics.irt.model.*;
import com.itemanalysis.psychometrics.quadrature.UserSuppliedQuadratureRule;
import com.itemanalysis.psychometrics.data.VariableName;
import java.io.*; import java.nio.file.*; import java.util.*;

public class Fit4PL {
  public static void main(String[] a) throws Exception {
    List<String> g = Files.readAllLines(Paths.get(a[1])); g.remove(0);
    double[] x = new double[g.size()], w = new double[g.size()];
    for (int i = 0; i < g.size(); i++) { String[] s = g.get(i).split(","); x[i] = Double.parseDouble(s[0]); w[i] = Double.parseDouble(s[1]); }
    String header = Files.readAllLines(Paths.get(a[0])).get(0); int n = header.split(",").length;
    ItemResponseVector[] data = new ItemResponseFileSummary().getResponseVectors(new File(a[0]), 0, n, true);
    ItemResponseModel[] irm = new ItemResponseModel[n];
    for (int j = 0; j < n; j++) { Irm4PL m = new Irm4PL(1.0, 0.0, 0.05, 0.95, 1.0); m.setName(new VariableName("M" + (j + 1))); irm[j] = m; }
    irm = new StartingValues(data, irm).computeStartingValues();
    MarginalMaximumLikelihoodEstimation mmle = new MarginalMaximumLikelihoodEstimation(data, irm, new UserSuppliedQuadratureRule(x, w));
    final StringBuilder log = new StringBuilder();
    mmle.addEMStatusListener(e -> log.append(e.getStatus()).append('\n'));
    long t0 = System.nanoTime(); mmle.estimateParameters(1e-6, 2000); double secs = (System.nanoTime() - t0) / 1e9;
    try (PrintWriter out = new PrintWriter(a[2])) {
      out.println("item,a,b,c,d");
      for (int j = 0; j < n; j++) { Irm4PL m = (Irm4PL) irm[j];
        out.println(header.split(",")[j] + "," + m.getDiscrimination() + "," + m.getDifficulty() + "," + m.getGuessing() + "," + m.getSlipping()); }
    }
    Files.write(Paths.get(a[2] + ".log"), (log.toString() + "seconds=" + secs + "\n").getBytes());
  }
}
