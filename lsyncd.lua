settings {
   nodaemon   = true
}
sync {
    default.rsyncssh,
    source  = ".",
    targetdir  = "/mnt/storage/users/lpigou/dsb3",
    exclude={"paths.yaml", ".*", "lsyncd.lua", "todo"},
    host='lpigou@paard.local',
    delete=false,
    delay=0.5,
    rsync = {
        archive = true,
        verbose=true,
        perms=false,
        times=false,
        _extra={}
    },
}
