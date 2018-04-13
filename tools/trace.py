from stacktracer import trace_start, trace_stop

if __name__ == '__main__':
    trace_start("trace.html", interval=5, auto=True)
    from cvlab import main
    main()
    trace_stop()
