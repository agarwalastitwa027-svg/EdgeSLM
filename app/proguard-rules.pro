# Native methods must survive shrinking/obfuscation - JNI resolves by mangled name.
-keepclasseswithmembernames class * {
    native <methods>;
}
