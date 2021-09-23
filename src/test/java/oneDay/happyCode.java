package oneDay;

import utils.ListNode;

import java.util.*;


class happyCode {
    //1893
//    差分数组diff表示相邻格之间，是否被覆盖的变化量。
//    diff[i]++,代表在i位置上有新的覆盖
//    若覆盖到j结束了呢？此时j依然是覆盖，但是j+1不在覆盖状态，所以在j+1处 -1；
//    即diff[j+1]--;

    public boolean isCovered(int[][] ranges, int left, int right) {
        int[] diff = new int[55];
        for(int[] a : ranges){
            diff[a[0]] ++ ;
            diff[a[1]+1] --;
        }

        //diff 前n项和
        for (int i = 1; i < diff.length; i++) {
            diff[i] = diff[i] + diff[i-1];
        }

        for (int i = left; i <= right; i++) {
            if(diff[i] <= 0)  return false;
        }

        return true;

    }


    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummpy = new ListNode(1);
        dummpy.next = head;

        for (int i = 0; i < left - 1; i++) {
            dummpy = dummpy.next;
        }
        ListNode lLeft = dummpy;
        ListNode leftNode = dummpy.next;
        for(int i = 0 ; i < right - left + 1;i++){
            dummpy = dummpy.next;
        }
        ListNode rightNode = dummpy;
        ListNode rRight = dummpy.next;

        lLeft.next = null;
        rightNode.next = null;
        ListNode newNode = reverse(leftNode);
        lLeft.next = newNode;
        for (int i = 0; i < right - left; i++) {
            newNode = newNode.next;
        }
        newNode.next = rRight;
        return head;


    }
    public ListNode reverse(ListNode head){
        ListNode dummpy = null;
        while(head != null){
            ListNode temp = head.next;
            head.next = dummpy;
            dummpy = head;
            head = temp;
        }
        return dummpy;
    }

    //public static String maximumTime(String time) {
//        char[] ch = time.toCharArray();
//        if(ch[0] == '?'){
//            if(ch[1] >= '4' && ch[1] <= '9'){
//                ch[0] = '1';
//            }else{
//                ch[0] = '2';
//            }
//        }
//
//        if(ch[1] == '?'){
//
//            ch[1] = ch[0] == '2' ? '3' : '9';
//        }
//
//        if(ch[3] == '?'){
//            ch[3] = '5';
//        }
//        if(ch[4] == '?'){
//            ch[4] = '9';
//        }
//
//        return new String(ch);
//
//    }

    //最大公约数
    public static int gcd(int a, int b){
        if(b == 0) return a;
        return gcd(b,a%b);
    }
    HashMap<Integer,TreeNode> map = new HashMap<>();
    ArrayList<Integer> res = new ArrayList<>();

    public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
        parentNode(root);
        findKson(target,k);
        findKparent(target,k);
        return res;

    }


    public void findKson(TreeNode target , int k ){
        if(target == null) return;
        if(k == 0) {
            res.add(target.val);
            return;
        }


        findKson(target.left,k-1);
        findKson(target.right,k-1);

    }

    public void findKparent(TreeNode target , int k ){
        if(map.get(target.val) == null) return;
        if(k == 0) {
            res.add(target.val);
            return;
        }


        TreeNode parent = map.get(target.val);
        findKparent(parent, k - 1);
        if (parent.left == target) {
            findKson(parent.right, k - 2);
        }
        if (parent.right == target) {
            findKson(parent.left, k - 2);
        }

    }



    public void parentNode(TreeNode root){
        if(root.left != null){
            map.put(root.left.val,root);
            parentNode(root.left);
        }
        if(root.right != null){
            map.put(root.right.val,root);
            parentNode(root.right);
        }
    }



//    public static void main(String[] args) {
////        String str = "1?:2?";
////        System.out.println(str.charAt(0) > '2' );
//        System.out.println(gcd(6,18));
//
//    }

    public List<Integer> pathInZigZagTree(int label) {
        int nodeValue = label;
        int[] temp = new int[label];
        int nodeLevel = 0;
        int i = 0;
        while(nodeValue >= 1){
            nodeValue /= 2;
            nodeLevel ++;
        }
        nodeValue = label;
        while(nodeValue >= 1){
            temp[i++] = nodeValue;

            nodeLevel--;
            int left = (int) Math.pow(2,nodeLevel-1);
            int right = (int) (Math.pow(2,nodeLevel) - 1);
            int nodeparent = nodeValue/2;
            nodeValue = right - (nodeparent - left);
        }

        ArrayList<Integer> res = new ArrayList<>();
        for (int j = temp.length-1; j >= 0; j--) {
            if(temp[j] != 0){
                res.add(temp[j]);
            }
        }

        return res;

    }

//    public int titleToNumber(String columnTitle) {
//        char[] ch = columnTitle.toCharArray();
//        int res = 0;
//        int len = ch.length;
//        for (int i = 0; i < ch.length; i++) {
//            int temp = (int) ((ch[i] - 'A' + 1) * Math.pow(26,len-- - 1));
//            res += temp;
//        }
//        return res;
//
//    }



    public static int[] kWeakestRows(int[][] mat, int k) {
        PriorityQueue<int[]> queue = new PriorityQueue<>( (int[] o1, int[] o2) -> {
            if (o1[1] != o2[1]) {
                return o1[1]-o2[1];
            }
            else {
                return o1[0]-o2[0];
            }
        });

        int length = mat.length;

        int[] res = new int[k];

        for (int i = 0; i < length; i++) {
//            temp[0] = i;
//            temp[1] = getNumber(mat[i]);
//            new temp{}
            queue.add(new int []{i,getNumber(mat[i])}  );
        }
        for (int i = 0; i < k; i++) {
            res[i] = queue.poll()[0];
        }
        return res;

    }

    public static int getNumber(int[] num){

        for (int i = 0; i < num.length; i++) {
            if(i == num.length -1 && num[i] == 1) return num.length;
            if(num[i] == 0) return i;
        }
        return 0;
    }

//    public static void main(String[] args) {
////        int[][] temp = {{1,1,1,1,1},{1,0,0,0,0},{1,1,0,0,0},{1,1,1,1,0},{1,1,1,1,1}};
////        int k = 3;
////        int[] res = kWeakestRows(temp,k);
////        for(int a : res){
////            System.out.println(a);
////        }
//        System.out.println(9/2);
//
//    }

    public boolean circularArrayLoop(int[] nums) {
        // 无论从哪个点出发，最终必然进入环路，但这个环路是否是有效的循环，需要检查确定
        // 检查每个位置做起点，进入的环路是否是有效的循环，只要存在一个，即为true
        for (int i = 0; i < nums.length; i++) {
            if(isCircular(nums,i)) return true;
        }
        return false;

    }

    public boolean isCircular(int[] nums , int begin){
        //第一遍遍历
        int slow = begin;
        int fast = getNext(nums,getNext(nums,begin));
        while(fast != slow){
            fast = getNext(nums,getNext(nums,fast));
            slow = getNext(nums,slow);
        }
        //再来一遍
        int flag = nums[fast];

        int k = 1;
        //满足同向及k>1
        while(getNext(nums,slow) != fast){
            if(nums[getNext(nums,slow)] * flag < 0) return false;
            slow = getNext(nums,slow);
            k ++;
        }
        return k > 1;
    }
    //获得下一个index
    public int getNext(int[] nums , int cur){
        int step = nums[cur] % nums.length;
        int next = cur + step;
        return (next + nums.length) % nums.length;
    }
    int[] temp = new int[38];
    public int tribonacci(int n) {

        temp[0] = 0;temp[1] = 1;temp[2] = 1;
        if(temp[n] != 0 || n == 0) return temp[n];
        return temp[n] = tribonacci(n-1)+tribonacci(n-2) + tribonacci(n-3);


    }

    public int[] reversePrint(ListNode head) {
        ListNode temp = head;
        ArrayList<Integer> list = new ArrayList<>();
        while(temp != null){
            //list.add(temp.val);
            temp = temp.next;
        }
        Collections.reverse(list);
        return list.stream().mapToInt(Integer::intValue).toArray();

    }

    public ListNode reverseList(ListNode head) {
        ListNode dummpy = null;
        ListNode temp = head;

        while(temp != null){
            head = head.next;
            temp.next = dummpy;
            dummpy = temp;
            temp = head;
        }
        return dummpy;

    }

    public Node copyRandomList(Node head) {
        Node p = head;
        while(p != null){
            Node node = new Node(p.val);
            node.next = p.next;
            p.next = node;
            p = node.next;
        }

        p = head;
        while(p != null){
            if(p.random != null){
                p.next.random = p.random.next;

            }
            p = p.next.next;
        }

        p = head;
        Node dummpyNode = new Node(-1);
        Node cur = dummpyNode;
        while(p != null){
            cur.next = p.next;
            cur = cur.next;
            p.next = cur.next;
            p = p.next;

        }

        return dummpyNode.next;
    }

//    public boolean isPrefixString(String s, String[] words) {
//
//        String str = "";
//        int lenS = s.length();
//        for(String string : words){
//            str += string;
//            if(str.length() > lenS) return false;
//            else if(str.equals(s)) return true;
//        }
//        return false;
//
//    }

    public static int minStoneSum(int[] piles, int k) {
        PriorityQueue<Integer> queue = new PriorityQueue<>((a,b) -> b-a);
        for(int i : piles){
            queue.add(i);
        }
        for (int i = 0; i < k; i++) {
            int temp = queue.poll();
            temp -= temp/2;
            queue.add(temp);
        }
        int res = 0;
        while(!queue.isEmpty()){
            res += queue.poll();
        }
        return res;

    }

//    public static void main(String[] args) {
//        int[] res = {5,4,9}; int l = 2;
//        int i = minStoneSum(res,l);
//    }
//    输入：n = 12, primes = [2,7,13,19]
//    输出：32   队列最小值 * primes[i]
//    解释：给定长度为 4 的质数数组 primes = [2,7,13,19]，前 12 个超级丑数序列为：[1,2,4,7,8,13,14,16,19,26,28,32]
    // 1 .   2 7 13 19 (弹出1).   7 13 19 4 14 26 36 (弹出2) .  弹出4
    public int nthSuperUglyNumber(int n, int[] primes) {
        PriorityQueue<Long> queue = new PriorityQueue<>();
        HashSet<Long> set = new HashSet<>();
        queue.add(1L);
        set.add(1L);

        for (int i = 0; i < n; i++) {
            long pop = queue.poll();
            if(i == n-1 ) return (int)pop;
            for(int cur : primes){
                if (!set.contains(pop * cur)) {
                    set.add(pop * cur);
                    queue.add(pop * cur);
                }
            }
        }
        return -1;

    }

    public String reverseLeftWords(String s, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = n; i < s.length(); i++) {
            sb.append(s.charAt(i));
        }
        for (int i = 0; i < n; i++) {
            sb.append(s.charAt(i));
        }

        return sb.toString();

    }

    public int[] twoSum(int[] nums, int target) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = i+1; j < nums.length; j++) {
                if(nums[i] + nums[j] == target){
                    return new int[]{i,j};
                }
            }
        }
        return new int[]{};

    }

    public int numberOfArithmeticSlices(int[] nums) {

        //dp数组表示当前位置新增的数列个数
        int length = nums.length;
        if(length < 3) return 0;
//        int[] dp = new int[length];
//        dp[0] = 0; dp[1] = 0;
        int dp = 0;

        int diff = nums[1] - nums[0];
        int count = 0;

        for (int i = 2; i < length; i++) {
            if(nums[i] - nums[i-1] == diff){
                dp = dp + 1;
                count += dp;
            }else{
                diff = nums[i] - nums[i-1];
            }
        }
        return count;

    }

    public int findRepeatNumber(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for(int i : nums){
            if(set.contains(i)){
                return i;
            }
            set.add(i);
        }
        return 0;

    }


    public static int search(int[] nums, int target) {
        int len = nums.length;
        int l = 0 , r = len - 1;
        int index = 0, sum = 0;
        while(l <= r){
            int mid = l + (r - l)/2;
            if(nums[mid] == target){
                index = mid;
                sum = 1;
                break;
            }
            else if(nums[mid] > target){
                r = mid -1;
            }else if(nums[mid] < target){
                l = mid + 1;
            }
        }
        int temp = index;

        while(0 <= index && index <= len -1){
            if(nums[index - 1] == target ){
                sum ++;
            }
            index --;
        }
        index = temp;
        while(0 <= index && index <= len -1){
            if(nums[index + 1] == target ){
                sum ++;
            }
            index++;
        }

        return sum;


    }

    public int missingNumber(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            if(nums[i] != i) return i;
        }
        return 0;

    }

    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        int i = matrix.length - 1, j = 0;
        while(i >= 0 && j < matrix[0].length)
        {
            if(matrix[i][j] > target) i--;
            else if(matrix[i][j] < target) j++;
            else return true;
        }
        return false;

    }

    public int minArray(int[] numbers) {
        Arrays.sort(numbers);
        return numbers[0];
    }

//    public static void main(String[] args) {
//        String s = "loveleetcode";
//        char a = firstUniqChar(s);
//        System.out.println(a);
//    }

    public static char firstUniqChar(String s) {
        if(s.length() == 0) return ' ';
        int[][] res = new int[26][2];

        for (int i = 0; i < s.length(); i++) {
            char temp = s.charAt(i);
            for(int[] ch : res){
                if(ch[0] == temp - 'a' + 1){
                    ch[1] ++;
                    break;
                }else if(ch[0] == 0 ){
                    ch[0] =  temp - 'a' + 1;
                    ch[1] ++;
                    break;
                }else if (ch[0] != temp - 'a' + 1){
                    continue;
                }
            }
        }

        PriorityQueue<int[]> queue = new PriorityQueue<>((a,b) -> a[1] - b[1]);
        for(int[] temp : res){
            if(temp[0] != 0){
                queue.add(temp);
            }else{
                break;
            }
        }
        int[] temp = queue.poll();
        if(temp[1] == 1) return (char)(temp[0] -1 + 'a');
        else return ' ';

    }

    public int maxArea(int[] height) {
        int len = height.length;
        int l = 0 , r = len - 1;
        int res = 0;
        while(l < r){
            int area = Math.min(height[l] , height[r]) * (r - l);
            res = Math.max(res,area);

            if(height[l] > height[r]) {
                r --;
            }else{
                l ++;
            }
        }
        return res;
    }

    public int[] levelOrder(TreeNode root) {
        if(root == null) return new int[0];
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        List<Integer> res = new ArrayList<>();

        while(!queue.isEmpty()){
            TreeNode temp = queue.poll();
            res.add(temp.val);
            if(temp.left != null )  queue.add(temp.left);
            if(temp.right != null )  queue.add(temp.right);
        }


        //return res.stream().mapToInt(Integer::valueOf).toArray();
        return res.stream().mapToInt(Integer::valueOf).toArray();

    }

    public List<List<Integer>> levelOrder2(TreeNode root) {

        List<List<Integer>> list = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();

        if(root == null) return list;
        queue.add(root);

        while(!queue.isEmpty()){
            int size = queue.size();
            List<Integer> temp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                temp.add(cur.val);
                if(cur.left != null) queue.add(cur.left);
                if(cur.right != null) queue.add(cur.right);
            }
            list.add(temp);
        }

        return list;



    }

    public List<List<Integer>> levelOrder3(TreeNode root) {
        List<List<Integer>> list = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();

        if(root == null) return list;
        queue.add(root);

        while(!queue.isEmpty()){
            LinkedList<Integer> temp = new LinkedList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int depth = list.size();
                TreeNode cur = queue.poll();
                if( depth % 2 == 0){ //偶数层 正着遍历
                    temp.add(cur.val);
                }else{
                    temp.addFirst(cur.val);
                }
                if(cur.left != null) queue.add(cur.left);
                if(cur.right != null) queue.add(cur.right);
            }
            list.add(temp);
        }

        return list;


    }

    public static int longestPalindromeSubseq(String s) {
        int len = s.length();
        int[][] dp = new int[len][len];
        System.out.println(s);
        System.out.println();
        int count = 0;

        for (int i = len - 1; i >= 0 ; i--) {
            dp[i][i] = 1;

            for (int j = i + 1; j < len; j++) {
                count++;
                if(s.charAt(i) == s.charAt(j)){
                    dp[i][j] = dp[i+1][j-1] + 2;
                }else{
                    dp[i][j] = Math.max(dp[i+1][j],dp[i][j-1]);
                }

            }

            System.out.println(s.charAt(i)+"开始");
            for (int j = 0; j < len; j++) {
                //System.out.println("i = " + i);
                System.out.print(dp[i][j]);
            }
            System.out.println();
        }
        System.out.println("count:"+count);

        return dp[0][len-1];

    }

    public static void main(String[] args) {
        String s = "aabbc";
        int res = longestPalindromeSubseq(s);
        System.out.println(res);
    }

    public int numWays(int n) {
        if(n == 0) return 1;
        int[] dp = new int[n+1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i < n+1; i++) {
            dp[i] = (dp[i-1] + dp[i-2]) % 1000000007;
        }
        return dp[n];
    }



    public int translateNum(int num) {
        String str = String.valueOf(num);
        char c = str.charAt(0);
        int[] dp = new int[str.length() + 1];
        //
        //判断当前遍历到的数字（索引i-1）(对应n的dp值)能否和前一个数字（索引i-2）组合起来翻译
        dp[0] = 1;
        dp[1] = 1;

        for (int i = 2; i < str.length() + 1; i++) {
            if(str.charAt(i-2) == '1' || (str.charAt(i-2) == '2') && str.charAt(i-1) <= '5'){
                dp[i] = dp[i-1] + dp[i-2];
            }else{
                dp[i] = dp[i-1];
            }
            System.out.println(dp[i]);
        }
        return dp[str.length()];
    }






}

class CQueue {
    Stack<Integer> stack1;
    Stack<Integer> stack2;

    public CQueue() {
        stack1 = new Stack<>();
        stack2 = new Stack<>();
    }

    public void appendTail(int value) {
        stack1.push(value);
    }

    public int deleteHead() {
        while(!stack1.isEmpty()){
            stack2.push(stack1.pop());
        }
        int result;
        if(stack2.isEmpty()) return -1;
        else{
            result = stack2.pop();
            while(!stack2.isEmpty()){
                stack1.push(stack2.pop());
            }
        }
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        queue.remove(-1);
        return result;

    }

    public int maxSubArray(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        int res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            dp[i] = Math.max(nums[i] , nums[i] + dp[i-1]);
            res = Math.max(res,dp[i]);
        }
        return res;

    }

//    public int lengthOfLongestSubstring(String s) {
//        Map<Character, Integer> dic = new HashMap<>();
//        int res = 0, tmp = 0;
//        for(int j = 0; j < s.length(); j++) {
//            int i = dic.getOrDefault(s.charAt(j), -1); // 获取索引 i
//            dic.put(s.charAt(j), j); // 更新哈希表
//            tmp = tmp < j - i ? tmp + 1 : j - i; // dp[j - 1] -> dp[j]
//            res = Math.max(res, tmp); // max(dp[j - 1], dp[j])
//        }
//        return res;
//
//    }

    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int res = 0 , tmp = 0;
        for (int i = 0; i < s.length(); i++) {
            // i 为遍历到的 字符处  j为上一个一样的字符所在的位置
            //tmp 记录 最长子字符串的长度
            int j = map.getOrDefault(s.charAt(i),-1);
            map.put(s.charAt(i),i);
            if(tmp < i - j) tmp++;
            else tmp = i-j;

            res = Math.max(res,tmp);
        }
        return res;
    }

    public static int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;

        int[][] dp = new int[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if(i == 0 && j == 0) dp[i][j] = grid[0][0];
                else if(i == 0 ) dp[i][j] = grid[i][j] + dp[i][j-1];
                else if(j == 0) dp[i][j] = grid[i][j] + dp[i-1][j];
                else dp[i][j] = Math.min(dp[i-1][j],dp[i][j-1]) + grid[i][j];
                System.out.print(dp[i][j]);

            }
            System.out.println();
        }
        return dp[m-1][n-1];

    }


    public int solve(int[] a , int[] b , int[] c , int k){
        //开始前能不能赢
        boolean first = compare(a,c) >= k;
        boolean second = true;
        if(first) return 0;

        int zeroB = 0;
        for(int i : b){
            if(i == 0) zeroB++;
        }

        if(!first && zeroB < k){
            return -1;
        }

        int res = 0;
        while(second){
            res++;
            for (int i = 0; i < a.length; i++) {
                a[i] += b[i];
            }
            if(compare(a,c) >= k) {
                second = false;
            }
        }
        return res;

    }

    //最多可以赢几把
    public int compare(int[] a , int[] b){
        Arrays.sort(a);
        Arrays.sort(b);
        int n = a.length;
        int i = 0 , j = 0,x = n-1 , y = n-1 ,cnt = 0;

        boolean last = true;

        while(last){
            if(x == i) last = false;
            if(a[x] > b[y]){
                x--;
                y--;
                cnt++;
            }else if(a[i] > b[j]){
                i++;
                j++;
                cnt++;
            }else if(a[i] < b[y]){
                i++;
                y--;
            }
        }
        return cnt;
    }

    public int findMaxConsecutiveOnes(int[] nums) {
        int res = 0;
        int tmp = 0;

        for (int i = 0; i < nums.length; i++) {
            if(nums[i] == 1){
                tmp ++;
            }else{
                tmp = 0;
            }

            res = Math.max(res,tmp);
        }
        return res;

    }

    public int findPoisonedDuration(int[] timeSeries, int duration) {

        int res = 0;

        for (int i = 0; i < timeSeries.length; i++) {
            if(i == timeSeries.length - 1){
                res += duration;
            }else if(timeSeries[i] + duration >= timeSeries[i+1]){
                res += timeSeries[i+1] - timeSeries[i];
            }
            else{
                res += duration;
            }
        }
        return res;

    }

    public int thirdMax(int[] nums) {
        Arrays.sort(nums);
        int tmp = nums[nums.length-1];
        int count = 1;
        //int res = 0;
        for (int i = nums.length - 2; i >= 0 ; i--) {
            if(tmp != nums[i]){
                tmp = nums[i];
                count ++;
            }

            if(count == 3){
                return tmp;
            }

        }
        return nums[nums.length-1];
    }

    public int maximumProduct(int[] nums) {
        int res = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i+1; j < nums.length; j++) {
                for (int k = j+1; k < nums.length; k++) {
                    res = Math.max(res,nums[i]*nums[j]*nums[k]);
                }
            }
        }
        return res;
    }

    public int[] findErrorNums(int[] nums) {
        int[] temp = new int[nums.length + 1];
        int[] res = new int[2];
        for (int i = 0; i < nums.length; i++) {
            temp[nums[i]] ++ ;
        }

        for (int i = 1; i < temp.length; i++) {
            if(temp[i] == 0) res[0] = temp[i];
            else if(temp[i] == 1) res[1] = temp[i];
        }
        return res;

    }

    public List<Integer> findDisappearedNumbers(int[] nums) {

        int[] tmp = new int[nums.length + 1];
        List<Integer> list = new ArrayList<>();

        for (int i = 0; i < nums.length; i++) {
            tmp[nums[i]] ++ ;
        }

        for (int i = 1; i < tmp.length; i++) {
            if(tmp[i] == 0) list.add(i);
        }
        return list;

    }

    public static List<Integer> findDuplicates(int[] nums) {
        int len = nums.length;
        ArrayList<Integer> list = new ArrayList<>();

        for(int i : nums){
            nums[ (i-1)%len] += len;
        }

        for (int i = 0; i < nums.length; i++) {
            if(nums[i] > 2 * len){
                list.add(i + 1);
            }
        }
        return list;

    }

    public int minMoves(int[] nums) {
        Arrays.sort(nums);
        int res = 0;

        for (int i = 1; i < nums.length; i++) {
            res += nums[i] - nums[0];

        }
        return res;

    }

    public static boolean checkPossibility0(int[] nums) {
        int dp = 0;
        for (int i = 1; i < nums.length ; i++) {
            if(nums[i] < nums[i-1]) dp ++ ;
            if(dp == 2) return false;
        }
        return true;
    }

    public static boolean checkPossibility(int[] nums) {

        if(nums.length == 1) return true;
        boolean flag = nums[0] > nums[1] ? false : true;

        for (int i = 1; i < nums.length; i++) {
            if(nums[i] < nums[i-1]){
                if(flag){
                    if(nums[i+1] >= nums[i-1]) nums[i] = nums[i+1]; //两种相对大小
                    else nums[i+1] = nums[i];

                    flag = false;
                }else{
                    return false;
                }
            }
        }

        return true;

    }

    public int[][] imageSmoother(int[][] img) {
        int m = img.length , n = img[0].length;
        int[][] smooth = new int[m][n];

        int[][] directions = {{-1,-1},{-1,0},{-1,1},{0,1},{0,-1},{1,-1},{1,0},{1,1}};

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int count = 1 , sum = img[i][j];
                for(int[] direction : directions){
                    int x = i + direction[0] , y = j + direction[1];
                    //int count = 1 , sum = img[i][j];
                    if(x >= 0 && y >= 0 && x < m && y < n){
                        sum += img[x][y]; count ++;
                    }
                }
                smooth[i][j] = sum / count;
            }
        }
        return smooth;

    }

    public int maxCount0(int m, int n, int[][] ops) {
        int[][] res = new int[m][n];
        int count = 0;
        for(int[] op : ops){
            for (int i = 0; i < op[0]; i++) {
                for (int j = 0; j < op[1]; j++) {
                    res[i][j] += 1;
                }
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if(res[i][j] == res[0][0]) count++;
            }
        }
        return count;

    }

    public int maxCount(int m, int n, int[][] ops) {

        for(int[] op : ops){
            m = Math.min(m,op[0]);
            n = Math.min(n,op[1]);
        }
        return m * n ;

    }

    public int countBattleships(char[][] board) {
        int count = 0;
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if(board[i][j] == 'X'){
                    if((i == 0 && j == 0) || (i == 0 && board[i][j-1] != 'X')
                    || (j == 0 && board[i-1][j] != 'X') || (board[i][j-1] != 'X' && board[i-1][j] != 'X')){
                        count++;
                    }
                }
            }

        }

        return count;

    }

    public int maxRotateFunction(int[] nums) {
        //直接模拟 超时

        int n = nums.length;
        int sum = 0 ;
        int ans = 0;

        for (int i = 0; i < n; i++) {
            sum += i;
            ans += i * nums[i];
        }

        int res = ans;

        for (int i = 1; i < n; i++) {
            int ansT = ans + sum - n * nums[n-i];
            res = Math.max(res,ansT);
            ans = ansT;
        }

        return res;

    }

    public int[][] matrixReshape(int[][] mat, int r, int c) {

        if(mat.length * mat[0].length != r * c ) return mat;
        int[][] res = new int[r][c];

        int x = 0 , y = 0;

        for(int[] i : mat){
            for(int j : i){
                if(y >= c){
                    x ++;
                    y = 0;
                }
                res[x][y++] = j;
            }
        }
        return res;


    }

    //
    public void rotate(int[][] matrix) {
        //目标 ： [i,j] -> [j, n-i-1]
        //第一步 沿主斜轴对称翻折 [i,j] -> [j,i]
        //第二步 沿中间竖线对称  [j,i] -> [j,n-i-1]

        int n = matrix.length;
        // step 1
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
        }

        int mid = n >> 1;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < mid; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[i][n - j - 1];
                matrix[i][n - j - 1] = tmp;
            }
        }

    }

    public void reverse (int[] nums){
        int l = nums.length - 1;
        int tmp = nums[l];
        while(l > 0){
            nums[l] = nums[l-1];l--;
        }
        nums[0] = tmp;
    }

    public static void gameOfLife(int[][] board) {
        int m = board.length;
        int n = board[0].length;
        int[][] copy = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int num = getNum(board,i,j);
                System.out.print(num);
                if(board[i][j] == 1 && num < 2){
                    copy[i][j] = 0;
                }else if(board[i][j] == 1 && (num == 2 || num == 3)) {
                    copy[i][j]=1;
                }else if(board[i][j] == 1 && num > 3){
                    copy[i][j] = 0;
                }else if(board[i][j] == 0 && num == 3){
                    copy[i][j] = 1;
                }else {
                    copy[i][j] = board[i][j];
                }
            }
            System.out.println();
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] = copy[i][j];
            }
        }

    }

    public static int getNum(int[][] nums , int i ,int j){
        int count = 0;
        int[][] directions = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
        for(int[] direction : directions){
            int x = i + direction[0];
            int y = j + direction[1];
            if(x >= 0 && x < nums.length && y >= 0 && y < nums[0].length){
                if(nums[x][y] == 1) count++;
            }
        }
        return count;

    }


//    public static void main(String[] args) {
////        int[][] board = {{0,1,0},{0,0,1},{1,1,1},{0,0,0}};
////        gameOfLife(board);
////        System.out.println();
////        for (int i = 0; i < board.length; i++) {
////            for (int j = 0; j < board[0].length; j++) {
////                System.out.print(board[i][j]);
////            }
////            System.out.println();
////        }
//        String str = "FlaG";
//
//
//    }

    public static boolean detectCapitalUse(String word) {
        int n = word.length();
        int[] arr = new int[n];
        int num = 0;
        for (int i = 0; i < n; i++) {
            if(word.charAt(i) < 'a'){
                arr[i] = 1;
                num ++ ;
            }
        }


        if(num == n || (arr[0] == 1 && num == 1) || num == 0){
            return true;
        }
        return false;

    }

    public static boolean isPalindrome(String s) {
        int left = 0, right = s.length() - 1;
        while (left < right) {
            char a = s.charAt(left), b = s.charAt(right);
            if (!((a >= 'A' && a <= 'Z') || (a >= 'a' && a <= 'z') || (a >= '0' && a <= '9'))) {
                left++;
                continue;
            }
            if (!((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9'))) {
                right--;
                continue;
            }
            // 统一化为小写
            if (a >= 'A' && a <= 'Z') a += 32;
            if (b >= 'A' && b <= 'Z') b += 32;
            if (a != b){
                return false;
            }
            right--;
            left++;
        }
        return true;
    }


    public String longestCommonPrefix(String[] strs) {
        if(strs.length == 0) return "";
        String res = strs[0];

        for (int i = 1; i < strs.length; i++) {
            int j = 0;
            for(;j<strs[i].length() && j < res.length();j++){
                if(res.charAt(j) != strs[i].charAt(j)){
                    break;
                }else continue;
            }
            res = res.substring(0,j);

        }
        return res;
    }

    public int countSegments1(String s) {
        if(s.length() == 0) return 0;
        String str = s.trim();
        if (str.equals("")) {
            return 0;
        }
        String[] arr = str.split("\\s+");
        return arr.length;

    }
    public int countSegments(String s) {
        if(s.length() == 0) return 0;
        String str = s.trim();
        int count = 0;
        for (int i = 0; i < str.length(); i++) {
            if(str.charAt(i) != ' ' && str.charAt(i+1) == ' '){
                count ++;
            }
        }
        return count;

    }
    public int lengthOfLastWord0(String str) {
        String s = str.trim();
        String[] arr = s.split("\\s+");
        return arr[arr.length-1].length();

    }

    public int lengthOfLastWord(String str) {
        String s = str.trim();
        int count = 0;
        for (int i = s.length()-1; i >= 0; i--) {
            if(s.charAt(i) == ' ') break;
            else count++;
        }
        return count;

    }

    public String reverseWords(String s) {
        String str = s.trim();
        String[] arr = str.split("\\s+");
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < arr.length; i++) {
            String tmp = reverse(arr[i]);
            sb.append(tmp);
            if( i != str.length() - arr[arr.length-1].length()){
                sb.append(' ');
            }
        }
        return sb.toString();

    }

    public String reverse(String s){
        StringBuilder sb = new StringBuilder();
        for (int i = s.length()-1; i >= 0; i--) {
            sb.append(s.charAt(i));
        }
        return sb.toString();
    }



    public String reverseWords1(String str) {
        String s = str.trim();
        String[] arr = s.split("\\s+");
        StringBuilder sb = new StringBuilder();
        for (int i = arr.length-1; i >= 0 ; i--) {
            sb.append(arr[i]);
            if(i != 0){
                sb.append(' ');
            }
        }
        return sb.toString();

    }

    public int firstUniqChar(String s) {
        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i),map.getOrDefault(s.charAt(i) , 0) + 1);
        }
        for (int i = 0; i < s.length(); i++) {
            if(map.get(s.charAt(i)) == 1){
                return i;
            }
        }
        return -1;
    }

    public char findTheDifference(String s, String t) {
        int[] arr = new int[26];
        for (int i = 0; i < t.length(); i++) {
            char tmp = t.charAt(i);
            arr[tmp-'a']++;
        }
        for (int i = 0; i < s.length(); i++) {
            char tmp = s.charAt(i);
            arr[tmp-'a']--;
        }
        for (int i = 0; i < arr.length; i++) {
            if(arr[i] == 1){
                return (char)(i+'a');
            }
        }
        return ' ';
    }

    public List<List<String>> groupAnagrams0(String[] strs) {

        int len = strs.length;
        List<List<String>> res = new ArrayList<>();
        String[] strings = new String[len];

        for (int i = 0; i < len; i++) {
            byte[] bytes = strs[i].getBytes();
            Arrays.sort(bytes);
            strings[i] = new String(bytes);
        }

        for (int i = 0; i < len; i++) {
            ArrayList<String> strings1 = new ArrayList<>();

            if(strings[i] != null) {
                strings1.add(strs[i]);
                for (int j = i + 1; j < len; j++) {

                    if (strings[i].equals(strings[j])) {
                        strings1.add(strs[j]);
                        strings[j] = null;
                    }
                }
            }
            if(strings1.size() != 0){
                res.add(strings1);
            }
        }
        return res;

    }


    public List<List<String>> groupAnagrams(String[] strs) {
        int len = strs.length;

        Map<String, List<String>> map = new HashMap<>();

        for(int i = 0; i < len; i++){
            char[] cs = strs[i].toCharArray();
            Arrays.sort(cs);
            String key = String.valueOf(cs);

            if (!map.containsKey(key))
                map.put(key, new ArrayList<>());

            map.get(key).add(strs[i]);
        }
        return new ArrayList<>(map.values());
    }

    public String originalDigits(String s) {
        int[] count = new int[26];
        int len = s.length();
        for (int i = 0; i < len; i++) {
            count[s.charAt(i)-'a']++;
        }

        int[]  res = new int[10];
        // letter "z" is present only in "zero"
        res[0] = count['z'-'a'];
        // letter "w" is present only in "two"
        res[2] = count['w'-'a'];
        // letter "u" is present only in "four"
        res[4] = count['u'-'a'];
        // letter "x" is present only in "six"
        res[6] = count['x'-'a'];
        // letter "g" is present only in "eight"
        res[8] = count['g'-'a'];
        // letter "h" is present only in "three" and "eight"
        res[3] = count['h'-'a'] - res[8];
        // letter "f" is present only in "five" and "four"
        res[5] = count['f'-'a'] - res[4];
        // letter "s" is present only in "seven" and "six"
        res[7] = count['s'-'a'] - res[6];
        // letter "i" is present in "nine", "five", "six", and "eight"
        res[9] = count['i'-'a'] - res[5] - res[6] - res[8];
        // letter "n" is present in "one", "nine", and "seven"
        res[1] = count['n'-'a'] - res[7] - 2 * res[9];

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < res.length; i++) {
            for (int j = 0; j < res[i]; j++) {
                sb.append(i);
            }
        }
        return sb.toString();


    }

    public boolean judgeCircle(String moves) {
        int up = 0 , down = 0;
        int left = 0 , right = 0;
        int len = moves.length();
        for (int i = 0; i < len; i++) {
            if(moves.charAt(i) == 'U') up ++;
            else if(moves.charAt(i) == 'D') down++;
            else if(moves.charAt(i) == 'L') left++;
            else if(moves.charAt(i) == 'R') right++;
        }

        if(up == down && left == right){
            return true;
        }
        return false;

    }

    public int countBinarySubstrings(String s) {

        ArrayList<Integer> list = new ArrayList<>();

        int left = 0, right = s.length();

        while(left < right){
            int count = 0;
            char ch = s.charAt(left);
            while(left < right && s.charAt(left) == ch){
                left++; count++;
            }
            list.add(count);
        }

        int res = 0;
        for (int i = 0; i < list.size() - 1; i++) {
            res += Math.min(list.get(i),list.get(i+1));
        }
        return res;

    }

    public String getHint(String secret, String guess) {
        StringBuilder sb = new StringBuilder();

        int countA = 0, len = secret.length();
        int[] arrSecret = new int[10] , arrGuess = new int[10];
        for (int i = 0; i < len; i++) {
            arrSecret[secret.charAt(i)]++;
            arrGuess[guess.charAt(i)]++;
            if(secret.charAt(i) == guess.charAt(i)){
                countA++;
            }
        }
        sb.append(countA);sb.append('A');
        int countB = 0;
        for (int i = 0; i < 10; i++) {
            if(arrGuess[i] != 0 && arrSecret[i] != 0){
                for (int j = 0; j < Math.min(arrGuess[i],arrSecret[i]); j++) {
                    countB++;
                }
            }
        }
        countB -= countA;
        sb.append(countB);sb.append('B');
        return sb.toString();


    }

    public List<String> fizzBuzz(int n) {
        List<String> list = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            if(i%3 == 0 && i%5 == 0) list.add("FizzBuzz");
            else if(i % 5 == 0) list.add("Buzz");
            else if(i % 3 == 0) list.add("Fizz");
            else list.add(String.valueOf(i));
        }
        return list;
    }

    public String[] findRelativeRanks(int[] score) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int len = score.length;
        for (int i = 0; i < len; i++) {
            map.put(score[i],i);
        }
        Arrays.sort(score);
        String[] str = new String[len];
        for (int i = len-1; i >= 0 ; i--) {
            int tmp = map.get(score[i]);
            if(i == len-1) str[tmp] = "Gold Medal";
            else if(i == len -2) str[tmp] = "Silver Medal";
            else if(i == len - 3) str[tmp] = "Bronze Medal";
            else str[tmp] = String.valueOf(len-i);
        }
        return str;
    }

    public int[] runningSum(int[] nums) {
        int len = nums.length;
        if(len == 0) return new int[0];
        int[] res = new int[len];
        res[0] = nums[0];

        for (int i = 1; i < len; i++) {
            res[i] = res[i-1] + nums[i];
        }

        return res;

    }

    public int sumOddLengthSubarrays(int[] arr) {
        int len = arr.length;
        if(len % 2 != 0){
            return (len + 1) * (len / 2 + 1) / 2;
        }else if(len % 2 == 0){
            return (len + 2) * (len / 2) /2;
        }
        return 0;

    }
    //solution 类测试
//    public static void main(String[] args) {
//        int[] arr = {4,8,6};
//        int[] subsum = new int[arr.length];
//        subsum[0] = arr[0];
//
//        for (int i = 1; i < arr.length; i++) {
//            subsum[i] = subsum[i-1] + arr[i];
//        }
//
//        int random = (int) (Math.random() * subsum[arr.length - 1]);
//        int l = 0 , r = arr.length -  1;
//
//        while( l < r){
//            int mid = (l + r)/2;
//            if(subsum[mid] == random){
//                System.out.println(mid + 1);
//            }else if(subsum[mid] > random){
//                r = mid;
//            }else{
//                l = mid + 1;
//            }
//        }
//        System.out.println(l);
//
//    }

    public int[] corpFlightBookings0(int[][] bookings, int n) {
        int[] res = new int[n];

        for(int[] booking : bookings){
            int tmp = booking[2];
            for (int i = booking[0]; i <= booking[1]; i++) {
                res[i] += tmp;

            }
        }
        return res;

    }

    public int[] corpFlightBookings(int[][] bookings, int n) {
        int[] res = new int[n];

        for(int[] booking : bookings){
            res[ booking[0] - 1] += booking[2];
            if(booking[1] < n){
                res[ booking[1] ] -= booking[2];
            }
        }
        for (int i = 1; i < n; i++) {
            res[i] += res[i-1];
        }
        return res;

    }

    public int compareVersion(String version1, String version2) {
        String[] str1 = version1.split("\\.");
        String[] str2 = version2.split("\\.");
        int len1 = str1.length , len2 = str2.length;
        int i = 0 , j = 0;

        while(i < len1 || j < len2){
            int m = 0 , n = 0;
            if(i < len1)  m = Integer.parseInt(str1[i++]);
            if(j < len2)  n = Integer.parseInt(str2[j++]);
            if(m != n){
                return m > n ? 1 : -1 ;
            }

        }
        return 0;

    }

    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode tmp = head;
        int len = 0;
        while(tmp != null){
            tmp = tmp.next;
            len++;
        }
        ListNode res = head;
        while(len > k){
            res = res.next;
            len--;
        }
        return res;

    }

    public static int calPoints(String[] ops) {

        ArrayList<String> res = new ArrayList<>();
        for(String str : ops){
            if(str.equals("C")) res.remove(res.size() - 1);
            else if(str.equals("D")){
                String s = res.get(res.size() - 1);
                int i = Integer.parseInt(s) * 2;
                res.add(String.valueOf(i));
            }else if(str.equals("+")){
                String s = res.get(res.size() - 1);
                String t = res.get(res.size() - 2);
                int i = Integer.parseInt(s) ;
                int j = Integer.parseInt(t);
                res.add(String.valueOf(i + j));
            }else {
                res.add(str);
            }

        }

        int sum = 0;
        for (int i = 0; i < res.size(); i++) {
            int tmp = Integer.parseInt(res.get(i));
            sum += tmp;
        }
        return sum;


    }

    public static String simplifyPath(String path) {
        String[] split = path.split("[/]+");
        ArrayList<String> list = new ArrayList<>();
        for(int i = 1; i < split.length;i ++){
            if(split[i].equals(".")) continue;
            else if(split[i].equals("..")) list.remove(list.size()-1);
            else list.add(split[i]);
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < list.size(); i++) {
            sb.append("/");
            sb.append(list.get(i));

        }
        return sb.toString();


    }

    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for(String str : tokens){
            if(str.equals("+")){
                int x = stack.pop() , y = stack.pop();
                stack.push(y + x);
            }else if(str.equals("-")){
                int x = stack.pop() , y = stack.pop();
                stack.push(y - x);
            }else if(str.equals("*")){
                int x = stack.pop() , y = stack.pop();
                stack.push(y * x);
            }else if(str.equals("/")){
                int x = stack.pop() , y = stack.pop();
                stack.push(y / x);
            }else{
                stack.push(Integer.parseInt(str));
            }
        }
        return stack.pop();

    }

    public static int calculate0(String s) {
        s = s.replaceAll(" ", "");
        Stack<Integer> stack = new Stack<>();
        int len = s.length();
        if(!( s.contains("+") || s.contains("-")  || s.contains("*")  || s.contains("/") )) return Integer.parseInt(s);
        for (int i = 0; i < len - 1; i++) {
            if(s.charAt(i) == '+'){
                stack.push(Integer.parseInt(String.valueOf(s.charAt(i+1))));
                i++;
            }else if(s.charAt(i) == '-'){
                stack.push(-Integer.parseInt(String.valueOf(s.charAt(i+1))));
                i++;
            }else if(s.charAt(i) == '*'){
                int x = stack.pop();
                stack.push(x * Integer.parseInt(String.valueOf(s.charAt(i+1))));
                i++;
            }else if(s.charAt(i) == '/'){
                int x = stack.pop();
                stack.push(x / Integer.parseInt(String.valueOf(s.charAt(i+1))));
                i++;
            }else{
                stack.push(Integer.parseInt(String.valueOf(s.charAt(i))));
            }
        }
        int sum = 0;
        while(!stack.isEmpty()){
            sum += stack.pop();
        }
        return sum;
    }

    public static int calculate(String s) {
        s = s.replaceAll(" ", "");
        String[] split = s.split("[-+*/]");
        Stack<Integer> stack = new Stack<>();
        int len = s.length();
        int count = 0;
        if(!( s.contains("+") || s.contains("-")  || s.contains("*")  || s.contains("/") )) return Integer.parseInt(s);
        stack.push(Integer.parseInt(split[0]));

        for (int i = 0; i < len; i++) {

            if(s.charAt(i) == '+'){
                count++;
                stack.push(Integer.parseInt(split[count]));
            }else if(s.charAt(i) == '-'){
                count++;
                stack.push(-Integer.parseInt(split[count]));
            }else if(s.charAt(i) == '*'){
                count++;
                stack.push(stack.pop() * Integer.parseInt(split[count]));
            }else if(s.charAt(i) == '/'){
                count++;
                stack.push(stack.pop() / Integer.parseInt(split[count]));
            }


        }

        int sum = 0;
        while(!stack.isEmpty()){
            sum += stack.pop();
        }
        return sum;
    }

    public static boolean isValid(String s) {

        LinkedList<Character> stack = new LinkedList<>();
        int len = s.length();
        for (int i = 0; i < len; i++) {
            if(s.charAt(i) == '(' || s.charAt(i) == '[' ||s.charAt(i) == '{' ){
                stack.add(s.charAt(i));
            }else{
                if(stack.isEmpty() || !check(stack.removeLast() , s.charAt(i))){
                    return false;
                }
                //是一对就pop左符号
                //stack.pop();
            }
        }
        if(stack.isEmpty()) return true;
        return false;


    }

    public static boolean check(char s , char t){
        if(s == '(' && t == ')') return true;
        else if(s == '[' && t == ']') return true;
        else if(s == '{' && t == '}') return true;
        return false;
    }

//    public static void main(String[] args) {
//        String str = "{[]}";
//        boolean valid = isValid(str);
//        System.out.println(valid);
//    }

    public int[] exclusiveTime(int n, List<String> logs) {
        Stack<Task> stack = new Stack<>();
        int[] res = new int[n];
        for(String log : logs){
            Task task = new Task(log.split(":"));
            if(task.start == true){
                stack.add(task);
            }else{

                Task pop = stack.pop();
                int duration = task.time - pop.time + 1;
                res[task.id] += duration;
                if(!stack.isEmpty()){
                    res[stack.peek().id] -= duration;
                }

            }
        }
        return res;

    }
    public int findMaximizedCapital(int k, int w, int[] profits, int[] capital) {
        int len = profits.length;
        ArrayList<int[]> list = new ArrayList<>();//capital-profits 键值对
        for (int i = 0; i < len; i++) {
            list.add(new int[]{capital[i],profits[i]});
        }
        Collections.sort(list, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });

        PriorityQueue<Integer> queue = new PriorityQueue<>( (a,b) -> b-a); //profits 从大到小
        int i = 0;
        while(k > 0){ //接k个活
            while(i < len && list.get(i)[0] <= w){
                queue.add(list.get(i)[1]);
                i++;
            }
            if(queue.isEmpty()) break;
            w += queue.poll();

        }
        return w;


    }

//    public static void main(String[] args) {
//        String[] words = {"What","must","be","acknowledgment","shall","be"};
//        int max = 16;
//        List<String> list = new ArrayList<>();
//        list = fullJustify(words,max);
//        System.out.println(list);
//    }

    public static List<String> fullJustify(String[] words, int maxWidth) {
        List<String> res = new ArrayList<>();
        ArrayDeque<String> duque = new ArrayDeque<>();
        for(String word : words){
            duque.add(word);
        }
        int nowWidth = 0;
        ArrayList<String> tmp = new ArrayList<>();
        while(!duque.isEmpty()){
            nowWidth += duque.peek().length();
            if (nowWidth > maxWidth){
                res.add(ListToString(tmp,maxWidth));
                tmp.clear();
                nowWidth = 0;
                continue;
            }
            tmp.add(duque.poll());
            nowWidth++;
        }
        //最后一行处理
        res.add(LastLineToString(tmp,maxWidth));
        return res;
    }

    public static String LastLineToString(ArrayList<String> list,int maxWidth){
        StringBuilder sb = new StringBuilder();
        int size = list.size();

        int nowlength = 0;

        for (int i = 0; i < size; i++) {
            sb.append(list.get(i));
            nowlength += list.get(i).length();
            if(nowlength == maxWidth) break;
            sb.append(" ");
            nowlength ++;
        }
        for (int i = 0; i < maxWidth - nowlength; i++) {
            sb.append(" ");
        }
        return sb.toString();
    }

    public static String ListToString(ArrayList<String> list,int maxWidth){

        StringBuilder sb = new StringBuilder();
        int size = list.size();
        int Stringlen = 0;
        for (String s : list) {
            Stringlen += s.length();
        }

        if(size == 1) { //一行只有一个单词
            String tmp = list.get(0);
            sb.append(tmp);
            for (int i = tmp.length(); i < maxWidth; i++) {
                sb.append(" ");
            }
        }else{  //一行多个单词
            int empty = maxWidth - Stringlen;
            int index = size - 1;
            int iposition = empty % index ;
            int number = empty / index;
            for (int i = 0; i < size; i++) {

                sb.append(list.get(i));
                if(i == size - 1){
                    break;
                }
                for (int j = 0; j < number; j++) {
                    sb.append(" ");
                }
                if(i < iposition){
                    sb.append(" ");
                }
            }
        }

        return sb.toString();
    }

    public static int chalkReplacer(int[] chalk, int k) {

        long sum = 0;
        int len = chalk.length;
        for(int i : chalk){
            sum += i;
        }
        if(k > sum){
            k = (int) (k % sum);
        }

        for (int i = 0; i < len; i++) {
            if(chalk[i] <= k)  k -= chalk[i];
            else return i;
        }

        return 0;

    }

    public int findIntegers(int n) {
        int[] dp = new int[32];
        dp[0] = 1;
        dp[1] = 2;
        for (int i = 2; i < 32; i++) {
            dp[i] = dp[i-1] + dp[i-2];
        }

        String str = convert(n);
        int res = 0;
        int len = str.length();
        // i = 0 对应二进制串 最高位
        for (int i = 0; i < len; i++) {
            if(str.charAt(i) == '0') continue;
            res += dp[len - i - 1];

            //有连续个1 不包含本身
            if(i >= 1 && str.charAt(i-1) == '1') return res;

        }
        //若遍历完成，则包含n本身
        return res+1;

    }
    // int 转为二进制串
    public String convert(int n){
        StringBuilder sb = new StringBuilder();
        while( n > 0){
            sb.append(n & 1);
            n >>= 1;
        }
        return sb.reverse().toString();
    }

    public int countQuadruplets(int[] nums) {
        int len = nums.length;
        int res = 0;

        for (int i = 0; i < len; i++) {
            for (int j = i + 1; j < len; j++) {
                for (int k = j + 1; k < len; k++) {
                    for (int l = k + 1; l < len; l++) {
                        if(nums[i] + nums[j] + nums[k] == nums[l])  res ++;
                    }
                }
            }
        }
        return res;

    }

    public static int numberOfWeakCharacters(int[][] properties) {

        int len = properties.length;
        int res = 0;
        HashSet<int[]> set = new HashSet<>();
        Arrays.sort(properties,new Comparator<int[]>(){
            @Override
            public int compare(int[]a,int[]b)
            {
                if(b[0]!=a[0])return a[0]-b[0];
                else return a[1]-b[1];
            }
        });

        for (int i = 1; i < len; i++) {
            int j = i;
            while(j>0 && properties[j][0] == properties[j-1][0]) j--;

            for (int k = 0; k < j; k++) {
                if(properties[i][1] > properties[k][1]) set.add(properties[k]);
            }
        }
        return set.size();
    }

//    public static void main(String[] args) {
//        int[][] arr = {{7,7},{1,2},{9,7},{7,3},{3,10},{9,8},{8,10},{4,3},{1,5},{1,5}};
//        int i = numberOfWeakCharacters(arr);
//        System.out.println(i);
//
//    }

    public static String findLongestWord(String s, List<String> dictionary) {
        String res = "";
        int sLen = s.length();
        for(String str : dictionary){
            int len = str.length();
            int j = 0;int i = 0;

            while(i!= len && j != sLen){
                if(str.charAt(i) == s.charAt(j)){
                    i++;
                }
                j++;
            }

            if(i == len && j <= sLen){
                if((str.length() > res.length())  || ((str.length() == res.length()) && str.compareTo(res) < 0)) res = str;
            }
        }
        return res;
    }

    public static void main(String[] args) {
        ArrayList<String> list = new ArrayList<>();

        list.add("aaa");list.add("aa");list.add("a");
        String a = "aaa";
        String s = findLongestWord(a,list);
        System.out.println(s);

    }

//    public static void main(String[] args) {
//        int[] arr = {3,4,1,2};
//        int k = 25;
//        int i = chalkReplacer(arr,k);
//        System.out.println(i);
//    }



}



class Task{
    int id;
    int time;
    boolean start;
    public Task(String[] split){
        this.id = Integer.parseInt(split[0]);
        this.time = Integer.parseInt(split[2]);
        this.start = "start".equals(split[1]);
    }
}

 class Codec {
    HashMap<Integer,String> map = new HashMap<>();
    int i = 0;

    // Encodes a URL to a shortened URL.
    public String encode(String longUrl) {
        map.put(i,longUrl);
        System.out.println("www.syl.com/"+i++);
        return "www.syl.com/"+i++;

    }

    // Decodes a shortened URL to its original URL.
    public String decode(String shortUrl) {
        return map.get(Integer.parseInt(shortUrl.replace("www.syl.com/","")));

    }
}

class MinStack {
    Stack<Integer> stack;
    PriorityQueue<Integer> queue = new PriorityQueue<Integer>();

    /** initialize your data structure here. */
    public MinStack() {
        stack = new Stack<>();

    }

    public void push(int x) {
        stack.push(x);
        queue.add(x);
    }

    public void pop() {
        int temp = stack.pop();
        queue.remove(temp);
    }

    public int top() {
        return stack.peek();
    }

    public int min() {
        return queue.peek();

    }
}

class NumArray {

    //int[] nums;
    int[] sums;

    public NumArray(int[] nums) {
        //this.nums = nums;
        sums = new int[nums.length];
        sums[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            sums[i] = sums[i-1] + nums[i];
        }
    }

    public int sumRange(int left, int right) {
        return sums[right] - sums[left];
    }
}

class Solution {

    int[] subsum;
    int len;

    public Solution(int[] w) {
        this.len = w.length;
        subsum = new int[len];
        subsum[0] = w[0];
        for (int i = 1; i < len; i++) {
            subsum[i] = subsum[i-1] + w[i];
        }

    }

    public int pickIndex() {
        int random = (int) (Math.random() * subsum[len-1]);
        int l = 0 , r = len - 1 ;

        while(l < r){
            //大于等于random的最小值
            int mid = (l + r )/2; //取低位
            if(subsum[mid] == random){
                return mid + 1;
            }else if(subsum[mid] > random){
                r = mid;
            }else{
                l = mid + 1;
            }
        }
        return r;
    }
}