import java.sql.*;

public class VulnSqlInjection {

    public ResultSet lookupUser(Connection conn, String username) throws SQLException {
        Statement stmt = conn.createStatement();
        String query = "SELECT * FROM users WHERE username = '" + username + "'";
        return stmt.executeQuery(query);
    }

    public static void main(String[] args) throws Exception {
        Connection conn = DriverManager.getConnection("jdbc:sqlite:test.db");
        new VulnSqlInjection().lookupUser(conn, args[0]);
    }
}
