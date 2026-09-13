// Joern CPGQL script: confirms an untrusted Java method parameter can reach
// a SQL-execution sink via a real data-flow path (CWE-89 class).
import java.nio.file.{Files, Path, Paths}

@main def main(cpgFile: String) = {
  val inputPath = Paths.get(cpgFile)
  val resolved: Path =
    if (Files.isDirectory(inputPath) && Files.exists(inputPath.resolve("cpg.bin"))) inputPath.resolve("cpg.bin")
    else inputPath

  importCpg(resolved.toString)

  println("=== Loaded CPG ===")
  println(s"Total AST nodes: ${cpg.all.size}")
  println(s"Total methods:   ${cpg.method.size}")

  val sink = cpg.call.name("executeQuery").argument.l
  val source = cpg.method("lookupUser").parameter.name("username").l

  println("\n=== Data-flow paths: source -> sink ===")
  val paths = sink.reachableByFlows(source).l
  if (paths.isEmpty) {
    println("  NO PATH FOUND")
  } else {
    println(s"  Found ${paths.size} path(s) confirming untrusted input reaches executeQuery:")
    paths.foreach { p =>
      p.elements.foreach(e => println(s"    -> ${e.code}  (${e.label})"))
      println("    ----")
    }
  }
}
