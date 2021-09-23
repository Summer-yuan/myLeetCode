package com.dailyQuestion.Sept;

public class splitListToParts725 {
    public static ListNode[] splitListToParts(ListNode head, int k) {
        ListNode dummpyNode = head;
        int len = 1;
        while(dummpyNode.next != null){
            len ++; dummpyNode = dummpyNode.next;
        }

        //dummpyNode.next = head;
        int num = len / k;
        int mod = len % k;
        ListNode[] res = new ListNode[k];
        for (int i = 0; i < k ; i++){

            ListNode node = new ListNode();
            node.next = head;
            ListNode pre = new ListNode();
            for (int j = 0; j < num && head != null; j++) {
                pre = head;
                head = head.next;
            }
            if(mod != 0){
                pre = head;
                head = head.next;
                mod --;
            }
            pre.next=null;
            res[i] = node.next;

        }
        return res;


    }

    public static void main(String[] args) {
        ListNode node1 = new ListNode(1);
        ListNode node2 = new ListNode(2);
        ListNode node3 = new ListNode(3,null);
        node1.next = node2;
        node2.next = node3;

        ListNode[] list = splitListToParts(node1,5);
        for (int i = 0; i < list.length; i++) {
            System.out.println(list[i].val);
        }

    }
}
