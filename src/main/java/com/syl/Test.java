package com.syl;

import java.lang.reflect.Method;

class Person{
    public void eat(){
        System.out.println("我吃");
    }

    public void eat(String s){
        System.out.println("我吃"+s);
    }
}
public class Test {
    public static void main(String[] args) throws Exception{
        //利用反射机制调用类的方法

        //1、获取类的字节码，字节码是根据源代码生成的，
        Class clazz = Class.forName("com.syl.Person");

        //2、利用反射机制创建一个对象，以下的api 就是 调用类的无参构造器来实例化对象的
        Object obj = clazz.newInstance();

        //3、反射出 字节码中的某个方法
        Method M = clazz.getDeclaredMethod("eat");
        Method m = clazz.getDeclaredMethod("eat",String.class);

        //4、利用反射机制调用方法
        //把m 所代表的方法，当做obj对象的方法来调用
        M.invoke(obj);
        m.invoke(obj,"大米饭");

    }
}
